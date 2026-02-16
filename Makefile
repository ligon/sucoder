SHELL := /bin/bash

.DEFAULT_GOAL := env-setup

# Defaults for recreating the human/agent environment.
AGENT_USER ?= coder
AGENT_GROUP ?= coder
HUMAN_USER ?= $(shell id -un)
CODER_CONFIG_DIR ?= $(HOME)/.sucoder
CODER_CONFIG ?= $(CODER_CONFIG_DIR)/config.yaml
SYSTEM_PROMPT ?= $(CODER_CONFIG_DIR)/system_prompt.org
CONFIG_SRC ?= config.example.yaml
SYSTEM_PROMPT_SRC ?= default_system_prompt.org
SKILLS_REPO ?= https://github.com/ligon/sucoder-skills.git
SKILLS_CLONE ?= $(HOME)/Projects/sucoder-skills
SUDO ?= sudo

.PHONY: help quick-start env-setup show-agent-user-commands create-agent-user config system-prompt skills-clone skills-update skills-link perms warmup poetry-ensure poetry-install test

help:
	@echo "Targets:"
	@echo "  quick-start     Create agent user + pip-install sucoder (zero-config, no config.yaml needed)"
	@echo "  env-setup       (default) Create config dir, seed config/prompt, clone skills, link them, set perms"
	@echo "  show-agent-user-commands  Print suggested commands to create coder user/group (requires root)"
	@echo "  create-agent-user         Create coder user/group (runs sudo/groupadd/useradd)"
	@echo "  config          Create ~/.sucoder and seed config.yaml if missing"
	@echo "  system-prompt   Copy default_system_prompt.org into ~/.sucoder if missing"
	@echo "  skills-clone    Clone sucoder-skills into $(SKILLS_CLONE) (or SKILLS_CLONE override)"
	@echo "  skills-update   git pull the skills clone"
	@echo "  skills-link     Symlink ~/.sucoder/skills -> skills clone if absent"
	@echo "  perms           Apply group-readable perms for $(AGENT_GROUP) on config + skills"
	@echo "  warmup          Run discovery checks (ls, sucoder skills-list, codex read catalog)"
	@echo "  poetry-ensure   Ensure Poetry is installed for $(AGENT_USER) (installs via curl if missing)"
	@echo "  poetry-install  Install project deps with Poetry"
	@echo "  test            Run pytest"

quick-start: create-agent-user
	pip install .

env-setup: create-agent-user poetry-ensure config system-prompt skills-clone skills-link perms

show-agent-user-commands:
	@echo "Run as root/admin to create the agent user and group:"
	@echo "  sudo groupadd -f $(AGENT_GROUP)"
	@echo "  id -u $(AGENT_USER) || sudo useradd -m -s /bin/bash -g $(AGENT_GROUP) -G $(AGENT_GROUP) $(AGENT_USER)"
	@echo "  sudo passwd -l $(AGENT_USER)   # lock password; use sudo/su for elevation when needed"
	@echo "  sudo chmod 755 /home/$(AGENT_USER)"
	@echo "  sudo usermod -aG $(AGENT_GROUP) $(HUMAN_USER)  # let human read/write shared files"
	@echo "  # then log out and back in for group membership to take effect"

create-agent-user:
	@echo "Ensuring agent group/user exist (requires sudo)..."
	$(SUDO) groupadd -f $(AGENT_GROUP)
	@if id -u $(AGENT_USER) >/dev/null 2>&1; then \
		echo "User $(AGENT_USER) already exists"; \
	else \
		echo "Creating $(AGENT_USER) with shell /bin/bash"; \
		$(SUDO) useradd -m -s /bin/bash -g $(AGENT_GROUP) -G $(AGENT_GROUP) $(AGENT_USER); \
		$(SUDO) passwd -l $(AGENT_USER); \
	fi
	$(SUDO) chmod 755 /home/$(AGENT_USER) || true
	@if id -nG $(HUMAN_USER) 2>/dev/null | grep -qw $(AGENT_GROUP); then \
		echo "$(HUMAN_USER) is already in group $(AGENT_GROUP)"; \
	else \
		echo "Adding $(HUMAN_USER) to group $(AGENT_GROUP)"; \
		$(SUDO) usermod -aG $(AGENT_GROUP) $(HUMAN_USER); \
		echo "Note: log out and back in (or run 'newgrp $(AGENT_GROUP)') for group membership to take effect"; \
	fi

config:
	@echo "Ensuring $(CODER_CONFIG_DIR) exists..."
	install -d -m 750 $(CODER_CONFIG_DIR)
	@if [ ! -f "$(CODER_CONFIG)" ]; then \
		echo "Seeding $(CODER_CONFIG) from $(CONFIG_SRC)"; \
		install -m 640 -T $(CONFIG_SRC) $(CODER_CONFIG); \
	else \
		echo "$(CODER_CONFIG) already exists; leaving in place"; \
	fi
	@echo "Review and edit $(CODER_CONFIG) for this host (users, mirror_root, skills, system_prompt)."

system-prompt:
	install -d -m 750 $(CODER_CONFIG_DIR)
	@if [ ! -f "$(SYSTEM_PROMPT)" ]; then \
		echo "Copying $(SYSTEM_PROMPT_SRC) -> $(SYSTEM_PROMPT)"; \
		install -m 640 -T $(SYSTEM_PROMPT_SRC) $(SYSTEM_PROMPT); \
	else \
		echo "$(SYSTEM_PROMPT) already exists; leaving in place"; \
	fi

skills-clone:
	@if [ -d "$(SKILLS_CLONE)/.git" ]; then \
		echo "Skills clone already present at $(SKILLS_CLONE)"; \
	else \
		echo "Cloning skills from $(SKILLS_REPO) into $(SKILLS_CLONE)"; \
		git clone $(SKILLS_REPO) $(SKILLS_CLONE); \
	fi

skills-update:
	@if [ -d "$(SKILLS_CLONE)/.git" ]; then \
		echo "Updating skills clone at $(SKILLS_CLONE)"; \
		git -C $(SKILLS_CLONE) pull --ff-only; \
	else \
		echo "No skills clone at $(SKILLS_CLONE); run 'make skills-clone' first" && exit 1; \
	fi

skills-link:
	install -d -m 750 $(CODER_CONFIG_DIR)
	@if [ -L "$(CODER_CONFIG_DIR)/skills" ] || [ -d "$(CODER_CONFIG_DIR)/skills" ]; then \
		echo "$(CODER_CONFIG_DIR)/skills already exists; leaving in place"; \
	else \
		echo "Linking $(CODER_CONFIG_DIR)/skills -> $(SKILLS_CLONE)"; \
		ln -s $(SKILLS_CLONE) $(CODER_CONFIG_DIR)/skills; \
	fi

perms:
	@echo "Setting permissions (may require sudo)..."
	$(SUDO) chgrp $(AGENT_GROUP) $(CODER_CONFIG_DIR) $(CODER_CONFIG) || true
	$(SUDO) chmod 750 $(CODER_CONFIG_DIR) || true
	$(SUDO) chmod 640 $(CODER_CONFIG) || true
	$(SUDO) chgrp -R $(AGENT_GROUP) $(SKILLS_CLONE) || true
	$(SUDO) chmod -R g+r $(SKILLS_CLONE) || true
	$(SUDO) chmod -R g-w $(SKILLS_CLONE) || true

warmup:
	ls -la $(CODER_CONFIG_DIR)
	ls -la $(CODER_CONFIG_DIR)/skills
	sucoder skills-list
	codex read $(CODER_CONFIG_DIR)/skills/SKILLS.md

poetry-ensure:
	@echo "Ensuring Poetry exists for $(AGENT_USER)..."
	@if $(SUDO) -u $(AGENT_USER) bash -lc 'command -v poetry >/dev/null 2>&1'; then \
		echo "Poetry already present for $(AGENT_USER)"; \
	else \
		echo "Installing Poetry for $(AGENT_USER) via install script"; \
		$(SUDO) -u $(AGENT_USER) bash -lc 'curl -sSL https://install.python-poetry.org | python3 -'; \
	fi
	@$(SUDO) -u $(AGENT_USER) bash -lc 'case ":$$PATH:" in *":$$HOME/.local/bin:"*) echo "PATH includes $$HOME/.local/bin";; *) echo "Note: add to PATH for $(AGENT_USER): export PATH=\"$$HOME/.local/bin:$$PATH\"";; esac'

poetry-install:
	poetry install

test:
	pytest
