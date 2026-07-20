# Vendored dependencies

Only dependencies that differ from their crates.io releases are vendored.

## `nice-plug`

Based on `nice-plug 0.1.9` (ISC). It exposes optional CLAP track-info and VST3
channel-context names to plugins. Hosts that do not implement those extensions
continue to work without track metadata. The upstream license is in
`nice-plug/LICENSE`.

## `nice-plug-egui`

Based on `nice-plug-egui 0.1.5` (ISC). It was migrated to `egui 0.35.0`,
`egui-baseview 0.6.0`, and `baseview 0.2.2` because those compatible versions
were not published together. The upstream license is in
`nice-plug-egui/LICENSE`.

## `baseview`

Based on `baseview 0.2.2` (MIT OR Apache-2.0). Its custom macOS `hitTest:`
override is disabled because its superclass dispatch recursively overflows the
stack on macOS 26. AppKit's default `NSView` hit testing remains active.

The patch is intentionally local and can be removed when an upstream release
fixes the issue. Upstream licenses are included in the crate directory.
