;;; validate.el --- Validate dream design Org artifacts -*- lexical-binding: t; -*-
;; Run from the SuCoder root: emacs --batch -Q -l docs/dream/validate.el
;; This checks document/export and native archive behavior, not dream runtime.

(require 'org)
(require 'ox-html)
(require 'org-archive)

(let* ((root (expand-file-name "../.." (file-name-directory load-file-name)))
       (default-directory root)
       (make-backup-files nil)
       (auto-save-default nil)
       (org-export-use-babel nil)
       (org-export-with-broken-links nil)
       (files (append '("docs/dream-charter.org" "docs/dream.org"
                        "docs/dream-ledger.org")
                      (directory-files-recursively "docs/dream" "\\.org\\'"))))
  (dolist (file files)
    (with-temp-buffer
      (setq buffer-file-name (expand-file-name file root))
      (setq default-directory (file-name-directory buffer-file-name))
      (insert-file-contents buffer-file-name)
      (org-mode)
      (let ((ids (make-hash-table :test 'equal))
            (tree (org-element-parse-buffer)))
        (org-element-map tree 'headline
          (lambda (heading)
            (let ((id (org-element-property :CUSTOM_ID heading)))
              (when id
                (when (gethash id ids) (error "Duplicate target %s in %s" id file))
                (puthash id t ids)))))
        (org-element-map tree 'link
          (lambda (link)
            (let ((type (org-element-property :type link))
                  (path (org-element-property :path link))
                  (search (org-element-property :search-option link)))
              (cond
               ((equal type "custom-id")
                (unless (gethash path ids)
                  (error "Missing internal target %s in %s" path file)))
               ((equal type "file")
                (let ((target (expand-file-name path default-directory)))
                  (unless (file-exists-p target)
                    (error "Missing linked file %s in %s" target file))
                  (when (and search (string-prefix-p "#" search))
                    (with-temp-buffer
                      (insert-file-contents target)
                      (unless (re-search-forward
                               (concat "^:CUSTOM_ID: +" (regexp-quote (substring search 1)) "[ \t]*$") nil t)
                        (error "Missing file target %s in %s" search target)))))))))))
      (org-export-as 'html nil nil t)
      (princ (format "Org export and targets: %s\n" file))))
  (let* ((scratch (make-temp-file "sucoder-dream-org-" t))
         (source-file (expand-file-name "topic.org" scratch))
         (archive-file (expand-file-name "archive.org" scratch)))
    (with-temp-file source-file
      (insert "#+todo: TODO NEXT ACTIVE WAITING REVIEW | DONE MOOT\n"
              "#+startup: logdone logdrawer\n#+archive: archive.org::\n"
              "* Topic\n** Talk :editorial:\n*** DONE Checked correction\n"
              "CLOSED: [2026-11-08 Sun 08:00]\n"
              ":PROPERTIES:\n:CUSTOM_ID: native-archive-check\n:END:\n"
              ":LOGBOOK:\n- State \"DONE\" from \"REVIEW\" [2026-11-08 Sun 08:00]\n:END:\n"
              "Keep the qualification and [[https://orgmode.org][source link]].\n"))
    (with-current-buffer (find-file-noselect source-file)
      (unless (equal org-done-keywords '("DONE" "MOOT"))
        (error "Unexpected terminal states"))
      (goto-char (org-find-property "CUSTOM_ID" "native-archive-check"))
      (org-back-to-heading t)
      (org-archive-subtree)
      (save-buffer))
    (with-current-buffer (find-file-noselect archive-file)
      (save-buffer)
      (goto-char (org-find-property "CUSTOM_ID" "native-archive-check"))
      (unless (and (equal (org-get-todo-state) "DONE")
                   (org-entry-get nil "CLOSED")
                   (equal (org-entry-get nil "ARCHIVE_ITAGS") "editorial")
                   (org-entry-get nil "ARCHIVE_FILE")
                   (org-entry-get nil "ARCHIVE_OLPATH")
                   (org-entry-get nil "ARCHIVE_TIME")
                   (save-excursion
                     (org-back-to-heading t)
                     (re-search-forward "Keep the qualification" nil t))
                   (save-excursion
                     (org-back-to-heading t)
                     (re-search-forward ":LOGBOOK:" nil t))
                   (save-excursion
                     (org-back-to-heading t)
                     (re-search-forward "https://orgmode.org" nil t)))
        (error "Native archive lost context, qualification, or logbook")))
    (princ "Native archive: custom target, completion/logbook, source, outline, time and inherited editorial tag preserved.\n"))
  (princ "Design artifact validation complete; no inference or repository test suite was run.\n"))
