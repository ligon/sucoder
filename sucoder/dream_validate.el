;;; dream_validate.el --- Parse fixture outputs without executing Org -*- lexical-binding: t; -*-
;; The coordinator supplies a bounded JSON object mapping wiki paths to text.
;; No file-local variables, links, source blocks, exports, or Babel are run.
(require 'json)
(require 'org)
(require 'org-element)

(defun sucoder-dream-link-path (source destination)
  "Normalize relative links as strings, without user or remote lookup."
  (let ((file-name-handler-alist nil))
    (if (or (string-prefix-p "~" destination)
            (file-name-absolute-p destination))
        destination
      (file-relative-name
       (expand-file-name destination
                         (file-name-directory (expand-file-name source "/"))) "/"))))

(let* ((inventory (member "--inventory" command-line-args-left))
       (reader (member "--reader" command-line-args-left))
       (history (member "--history" command-line-args-left))
       (inputs (json-read-from-string
                (with-temp-buffer
                  (insert-file-contents "/dev/stdin")
                  (buffer-string))))
       (result (make-hash-table :test 'equal))
       (details (make-hash-table :test 'equal))
       (read-details (make-hash-table :test 'equal))
       (links nil))
  (dolist (flag '("--inventory" "--reader" "--history"))
    (setq command-line-args-left (delete flag command-line-args-left)))
  (dolist (entry inputs)
    (let ((path (symbol-name (car entry)))
          (ids (make-hash-table :test 'equal))
          (headings nil)
          (keywords nil)
          (all-links nil))
      (with-temp-buffer
        (insert (cdr entry))
        ;; org-collect-keywords follows SETUPFILE during org-mode setup unless
        ;; the buffer is read-only. Keep source-directed file/network reads off.
        (setq buffer-read-only t)
        (let ((org-mode-hook nil)
              (org-todo-keywords (if inventory
                                     '((sequence "TODO" "NEXT" "ACTIVE" "WAITING" "REVIEW" "|" "DONE" "MOOT"))
                                   org-todo-keywords))
              (org-inhibit-startup t))
          (org-mode))
        (let ((tree (org-element-parse-buffer)))
          (when inventory
            (org-element-map tree 'keyword
              (lambda (keyword)
                (push `((key . ,(org-element-property :key keyword))
                        (value . ,(org-element-property :value keyword))) keywords))))
          (org-element-map tree 'headline
            (lambda (heading)
              (let* ((id (org-element-property :CUSTOM_ID heading))
                     (hash (secure-hash 'sha256
                            (encode-coding-string
                             (buffer-substring-no-properties
                              (org-element-property :begin heading)
                              (org-element-property :end heading))
                             'utf-8-unix))))
                (when inventory
                  (push `((target . ,id)
                          (title . ,(org-element-property :raw-value heading))
                          (level . ,(org-element-property :level heading))
                          (state . ,(org-element-property :todo-keyword heading))
                          (sha256 . ,hash)) headings))
                (when id
                  (when (gethash id ids)
                    (error "Duplicate custom target in %s" path))
                  (puthash id hash ids)))))
          (org-element-map tree 'link
            (lambda (link)
              (let ((type (org-element-property :type link))
                    (destination (org-element-property :path link))
                    (search (org-element-property :search-option link)))
                (when inventory
                  (push `((type . ,type)
                          (raw_path . ,destination)
                          (path . ,(if (equal type "file")
                                       (sucoder-dream-link-path path destination)
                                     destination))
                          (search . ,search)) all-links))
                (cond
                 ((equal type "custom-id")
                  (push (list path path destination) links))
                 ((and (equal type "file") search (string-prefix-p "#" search))
                  (push (list path
                              (sucoder-dream-link-path path destination)
                              (substring search 1)) links)))))))
        (when reader
          (let ((regions nil) (removed 0) (sections nil))
            (unless history
              (org-element-map (org-element-parse-buffer) 'drawer
                (lambda (drawer)
                  (when (equal (org-element-property :drawer-name drawer) "LOGBOOK")
                    (push (cons (org-element-property :begin drawer)
                                (org-element-property :end drawer)) regions))))
              (let ((inhibit-read-only t))
                (dolist (region (sort regions (lambda (a b) (> (car a) (car b)))))
                  (setq removed (+ removed (string-bytes (buffer-substring-no-properties (car region) (cdr region)))))
                  (delete-region (car region) (cdr region)))))
            (org-element-map (org-element-parse-buffer) 'headline
              (lambda (heading)
                (let ((target (org-element-property :CUSTOM_ID heading)))
                  (when target
                    (push `((target . ,target)
                            (begin . ,(string-bytes (buffer-substring-no-properties (point-min) (org-element-property :begin heading))))
                            (end . ,(string-bytes (buffer-substring-no-properties (point-min) (org-element-property :end heading))))) sections)))))
            (puthash path `((text . ,(buffer-string)) (logbook_bytes_omitted . ,removed)
                            (sections . ,(vconcat (nreverse sections)))) read-details)))
        (puthash path ids result)
        (puthash path `((targets . ,ids)
                        (headings . ,(vconcat (nreverse headings)))
                        (keywords . ,(vconcat (nreverse keywords)))
                        (links . ,(vconcat (nreverse all-links)))) details))))
  (dolist (link (unless reader links))
    ;; Existing links outside the changed set retain their original semantics;
    ;; the fixture's new links all resolve inside the supplied generated files.
    (let ((ids (gethash (nth 1 link) result)))
      (when (and ids (not (gethash (nth 2 link) ids)))
        (error "Missing custom target in %s" (car link)))))
  (princ (json-encode (cond (reader read-details) (inventory details) (t result)))))
