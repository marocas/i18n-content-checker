---
description: 'i18n: Use when creating, editing, or reviewing Payload CMS collections. Covers field definitions, localization, access control, hooks, and admin config.'
applyTo: 'src/collections/**'
---

# Payload Collection Conventions

## Collection Structure

Collections live in `src/collections/`. Simple collections are a single file; complex ones use a folder:

```
src/collections/
├── Categories.ts          # Simple collection (single file)
├── Media.ts
├── Pages/
│   ├── index.ts           # Collection config
│   └── hooks/             # Collection-specific hooks
│       └── revalidatePage.ts
└── Users/
    └── index.ts
```

## Field Rules

- Use `localized: true` on any user-facing text field (title, description, content)
- Prices are stored **in cents** (e.g. `1500` = 15.00)
- Use `slugField()` from `payload` for slug fields
- Use `@payloadcms/plugin-seo/fields` for SEO fields (`MetaTitleField`, `MetaDescriptionField`, etc.)

## Access Control

Import access functions from `src/access/`:

```typescript
import { authenticated } from '@/access/authenticated'
import { authenticatedOrPublished } from '@/access/authenticatedOrPublished'

access: {
  create: authenticated,
  delete: authenticated,
  read: authenticatedOrPublished,  // Public read for published content
  update: authenticated,
}
```

## Hooks

- Use `afterChange` hooks for revalidation (`revalidateDelete`, `revalidatePage`)
- Use `populatePublishedAt` from `@/hooks/populatePublishedAt` for publishable content
- Place collection-specific hooks in `<Collection>/hooks/`

## Admin Config

```typescript
admin: {
  defaultColumns: ['title', 'slug', 'updatedAt'],
  useAsTitle: 'title',  // Field used as document title in admin
}
```

## Draft/Preview Support

For content collections (pages, posts), enable:

- `versions.drafts: true`
- `livePreview` URL via `generatePreviewPath()`
- `preview` function for admin preview button

## Reference

See [Pages/index.ts](../../src/collections/Pages/index.ts) for a complete example with all patterns applied.
