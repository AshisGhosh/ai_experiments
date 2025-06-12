#!/bin/bash

echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Source cargo environment for current shell
source "$HOME/.cargo/env"

# Add cargo to PATH in .bashrc if not already present
if ! grep -q "source \"\$HOME/.cargo/env\"" "$HOME/.bashrc"; then
    echo "Configuring PATH in .bashrc..."
    echo 'source "$HOME/.cargo/env"' >> "$HOME/.bashrc"
fi

echo "Installing just..."
cargo install just

echo "Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --upgrade pip --system
uv pip install ruff --system

echo "Installation complete!"
echo "Please run 'source ~/.bashrc' or restart your terminal to ensure cargo is in your PATH"