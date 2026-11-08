;;; dream_validate.el --- Parse fixture outputs without executing Org -*- lexical-binding: t; -*-
;; The coordinator supplies a bounded JSON object mapping wiki paths to text.
;; No file-local variables, links, source blocks, exports, or Babel are run.
(require 'json)
(require 'org)
(require 'org-element)

(let* ((inputs (json-read-from-string
                (with-temp-buffer
                  (insert-file-contents "/dev/stdin")
                  (buffer-string))))
       (result (make-hash-table :test 'equal))
       (links nil))
  (dolist (entry inputs)
    (let ((path (symbol-name (car entry)))
          (ids (make-hash-table :test 'equal)))
      (with-temp-buffer
        (insert (cdr entry))
        ;; org-collect-keywords follows SETUPFILE during org-mode setup unless
        ;; the buffer is read-only. Keep source-directed file/network reads off.
        (setq buffer-read-only t)
        (let ((org-mode-hook nil)
              (org-inhibit-startup t))
          (org-mode))
        (let ((tree (org-element-parse-buffer)))
          (org-element-map tree 'headline
            (lambda (heading)
              (let ((id (org-element-property :CUSTOM_ID heading)))
                (when id
                  (when (gethash id ids)
                    (error "Duplicate custom target in %s" path))
                  (puthash id
                           (secure-hash 'sha256
                            (encode-coding-string
                             (buffer-substring-no-properties
                              (org-element-property :begin heading)
                              (org-element-property :end heading))
                             'utf-8-unix)) ids)))))
          (org-element-map tree 'link
            (lambda (link)
              (let ((type (org-element-property :type link))
                    (destination (org-element-property :path link))
                    (search (org-element-property :search-option link)))
                (cond
                 ((equal type "custom-id")
                  (push (list path path destination) links))
                 ((and (equal type "file") search (string-prefix-p "#" search))
                  (push (list path
                              (file-relative-name
                               (expand-file-name destination
                                                 (file-name-directory (expand-file-name path "/")))
                               "/")
                              (substring search 1)) links)))))))
        (puthash path ids result))))
  (dolist (link links)
    ;; Existing links outside the changed set retain their original semantics;
    ;; the fixture's new links all resolve inside the supplied generated files.
    (let ((ids (gethash (nth 1 link) result)))
      (when (and ids (not (gethash (nth 2 link) ids)))
        (error "Missing custom target in %s" (car link)))))
  (princ (json-encode result)))
