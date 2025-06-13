#!/bin/bash

# add just completion to .bashrc if not already present
if ! grep -q "_just_recipes()" ~/.bashrc; then
  cat << 'EOF' >> ~/.bashrc

_just_recipes() {
  local cur recipes
  cur=${COMP_WORDS[COMP_CWORD]}
  recipes=$(just --summary 2>/dev/null | tr ' ' '\n')
  COMPREPLY=( $(compgen -W "$recipes" -- "$cur") )
}

complete -F _just_recipes just
EOF
fi


# start bash
exec /bin/bash