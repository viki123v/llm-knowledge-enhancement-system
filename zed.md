# Zed Notes: LSP, Languages, Commands, and Actions

Research date: 2026-06-15.

## Core Mental Model

Zed commands shown in the Command Palette have labels like `zed: open settings` or `editor: restart language server`. In keymaps, the same commands use Rust-style action names like `zed::OpenSettings` or `editor::RestartLanguageServer`.

Zed's own docs say that commands of the form `zed: ...`, `editor: ...`, etc. are meant to be executed through the Command Palette. The Command Palette is opened with `cmd-shift-p` on macOS or `ctrl-shift-p` on Linux/Windows.

For keybindings, Zed exposes most functionality as actions. Keymap entries use the `namespace::ActionName` form, for example:

```json
[
  {
    "bindings": {
      "ctrl-a": "language_selector::Toggle"
    }
  }
]
```

## Important LSP/Language Actions

These are from Zed's official "All Actions" reference.

| Command Palette label | Keymap action | Meaning |
| --- | --- | --- |
| `language selector: toggle` | `language_selector::Toggle` | Opens the language selector modal for the current buffer. This is the key command when Zed detected the wrong language. |
| `lsp tool: toggle menu` | `lsp_tool::ToggleMenu` | Opens/toggles the language server tool menu. Useful for seeing/interacting with LSP-related tools for the active buffer. |
| `editor: restart language server` | `editor::RestartLanguageServer` | Restarts the language server for the current file. |
| `editor: stop language server` | `editor::StopLanguageServer` | Stops the language server for the current file. |
| `editor: cancel language server work` | `editor::CancelLanguageServerWork` | Cancels pending language server work. |
| `dev: open language server logs` | `dev::OpenLanguageServerLogs` | Opens the language server protocol logs viewer. |
| `zed: open log` | `zed::OpenLog` | Opens the general Zed log file. |
| `zed: open settings` | `zed::OpenSettings` | Opens the settings editor. |
| `zed: open settings file` | `zed::OpenSettingsFile` | Opens the raw user settings JSON file. |
| `zed: open project settings` | `zed::OpenProjectSettings` | Opens project-specific settings UI. |
| `zed: open project settings file` | `zed::OpenProjectSettingsFile` | Opens `.zed/settings.json` for the project. |
| `zed: open keymap` | `zed::OpenKeymap` | Opens the keymap editor. |
| `zed: open keymap file` | `zed::OpenKeymapFile` | Opens the raw user keymap JSON file. |
| `dev: open key context view` | `dev::OpenKeyContextView` | Shows active keybinding contexts; useful when a keybinding does not fire. |

## Answer to the Original LSP Scenario

If Zed detected the wrong file type/language and therefore started the wrong LSP, the practical manual fix is:

1. Run `language selector: toggle`.
2. Pick the correct language for the current buffer.
3. Zed should then use the language-server configuration attached to that language.
4. If needed, run `editor: restart language server`.
5. If diagnosing, run `lsp tool: toggle menu` or `dev: open language server logs`.

The important correction to my earlier answer: there are LSP-specific commands/actions. The most relevant ones are `lsp_tool::ToggleMenu`, `editor::RestartLanguageServer`, and `editor::StopLanguageServer`. However, Zed still fundamentally associates language servers with languages/buffers. Manually changing the buffer language is the normal way to recover from wrong detection.

Useful keymap example:

```json
[
  {
    "context": "Editor && mode == full",
    "bindings": {
      "cmd-k l": "language_selector::Toggle",
      "cmd-k r": "editor::RestartLanguageServer",
      "cmd-k s": "editor::StopLanguageServer",
      "cmd-k m": "lsp_tool::ToggleMenu"
    }
  }
]
```

On Linux/Windows, replace `cmd-` with `ctrl-` or whatever fits the local keymap.

## Configuring Language Servers

Language-specific settings go under `languages` in `settings.json`:

```json
{
  "languages": {
    "Python": {
      "tab_size": 4,
      "formatter": "language_server",
      "format_on_save": "on"
    }
  }
}
```

Language server settings go under top-level `lsp`:

```json
{
  "lsp": {
    "rust-analyzer": {
      "initialization_options": {
        "check": {
          "command": "clippy"
        }
      }
    }
  }
}
```

Zed distinguishes:

- `initialization_options`: sent during LSP startup; generally needs a language-server restart to apply.
- `settings`: served via LSP configuration requests; many servers use this style.
- `binary`: controls how Zed launches a server binary, including `path`, `arguments`, and `env`.

Example explicit binary:

```json
{
  "lsp": {
    "rust-analyzer": {
      "binary": {
        "ignore_system_version": false,
        "path": "/path/to/langserver/bin",
        "arguments": ["--option", "value"],
        "env": {
          "FOO": "BAR"
        }
      }
    }
  }
}
```

## Enabling/Disabling LSP

Language server support can be disabled per language:

```json
{
  "languages": {
    "Markdown": {
      "enable_language_server": false
    }
  }
}
```

This can live globally in `~/.config/zed/settings.json` or per project in `.zed/settings.json`.

## File Associations

Zed automatically detects file types, but project/user overrides are done with `file_types`:

```json
{
  "file_types": {
    "C++": ["c"],
    "TOML": ["MyLockFile"],
    "Dockerfile": ["Dockerfile*"]
  }
}
```

This is the persistent solution when detection is consistently wrong. For one-off/manual correction, prefer `language selector: toggle`.

## Troubleshooting Checklist

When an LSP is not attached, attached to the wrong language, or behaving incorrectly:

1. Run `language selector: toggle` and confirm the buffer language.
2. Run `lsp tool: toggle menu` to inspect active LSP tooling.
3. Run `editor: restart language server`.
4. Run `dev: open language server logs` for LSP protocol logs.
5. Run `zed: open log` for general Zed logs.
6. Check `languages` and `lsp` sections in user/project settings.
7. If detection is repeatedly wrong, add or adjust `file_types`.

## Sources

- Zed Command Palette: https://zed.dev/docs/command-palette
- Zed Keybindings and Actions: https://zed.dev/docs/key-bindings
- Zed All Actions: https://zed.dev/docs/all-actions
- Zed Configuring Languages / LSP: https://zed.dev/docs/configuring-languages
