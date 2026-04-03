;;; publish.el --- Build SuCoder documentation site  -*- lexical-binding: t; -*-
;;
;; Usage:
;;   emacs --batch -l docs/publish.el --funcall org-publish-all
;;
;; Exports .org files from the project root and docs/ to HTML in _site/.

(require 'ox-publish)
(require 'ox-html)

;; Derive paths from this script's location.
(defvar sucoder-root
  (file-name-directory
   (directory-file-name
    (file-name-directory (or load-file-name buffer-file-name)))))

(defvar sucoder-output (expand-file-name "_site" sucoder-root))

;; Shared HTML settings.
(setq org-html-validation-link nil
      org-html-head-include-scripts nil
      org-html-head-include-default-style nil
      org-html-head "<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" />"
      org-html-postamble t
      org-html-postamble-format
      '(("en" "<p class=\"postamble\">SuCoder &mdash; <a href=\"index.html\">Home</a> &middot; <a href=\"https://github.com/ligon/SuCoder\">GitHub</a></p>")))

;; Suppress interactive prompts and optional features during batch export.
(setq org-confirm-babel-evaluate nil)

;; Disable citation processing — citeproc is not available in bare Emacs.
;; The org-authoring-demo uses citations; it is excluded via :exclude below.

;; Disable source-block fontification (htmlize not available).
(setq org-html-htmlize-output-type nil)

(setq org-publish-project-alist
      `(
        ;; Root-level org files (README, prompts).
        ("sucoder-root-org"
         :base-directory ,sucoder-root
         :base-extension "org"
         :publishing-directory ,sucoder-output
         :publishing-function org-html-publish-to-html
         :recursive nil
         :with-toc t
         :section-numbers nil
         :html-head "<link rel=\"stylesheet\" type=\"text/css\" href=\"style.css\" />"
         :html-postamble t)

        ;; Docs subtree (excludes examples/ which needs citeproc).
        ("sucoder-docs-org"
         :base-directory ,(expand-file-name "docs" sucoder-root)
         :base-extension "org"
         :exclude "examples/"
         :publishing-directory ,(expand-file-name "docs" sucoder-output)
         :publishing-function org-html-publish-to-html
         :recursive t
         :with-toc t
         :section-numbers nil
         :html-postamble t)

        ;; Static assets (CSS, images).
        ("sucoder-static"
         :base-directory ,(expand-file-name "docs" sucoder-root)
         :base-extension "css\\|png\\|jpg\\|svg\\|gif"
         :publishing-directory ,(expand-file-name "docs" sucoder-output)
         :publishing-function org-publish-attachment
         :recursive t)

        ;; Copy CSS to site root so README.html can find it.
        ("sucoder-root-static"
         :base-directory ,(expand-file-name "docs" sucoder-root)
         :base-extension "css"
         :publishing-directory ,sucoder-output
         :publishing-function org-publish-attachment
         :recursive nil)

        ;; Umbrella project.
        ("sucoder" :components ("sucoder-root-org"
                                "sucoder-docs-org"
                                "sucoder-static"
                                "sucoder-root-static"))))

(provide 'publish)
;;; publish.el ends here
