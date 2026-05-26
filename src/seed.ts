import type { Payload } from 'payload'

export const seed = async (payload: Payload): Promise<void> => {
  payload.logger.info('— Seeding database...')

  // Create admin user
  const existingUsers = await payload.find({ collection: 'users', limit: 1 })
  if (existingUsers.totalDocs === 0) {
    await payload.create({
      collection: 'users',
      data: {
        email: 'admin@example.com',
        password: 'admin123',
      },
    })
    payload.logger.info('  ✓ Admin user created (admin@example.com / admin123)')
  }

  // Create media (placeholder image)
  let heroImage: number | undefined
  const existingMedia = await payload.find({ collection: 'media', limit: 1 })
  if (existingMedia.totalDocs === 0) {
    // Create a 1x1 pixel PNG buffer as placeholder
    const placeholderBuffer = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPj/HwADBwIAMCbHYQAAAABJRU5ErkJggg==',
      'base64',
    )
    const media = await payload.create({
      collection: 'media',
      data: { alt: 'Hero placeholder' },
      file: {
        data: placeholderBuffer,
        mimetype: 'image/png',
        name: 'placeholder.png',
        size: placeholderBuffer.length,
      },
    })
    heroImage = media.id
    payload.logger.info('  ✓ Placeholder media created')
  } else {
    heroImage = existingMedia.docs[0]?.id
  }

  // Create home page
  const existingPages = await payload.find({
    collection: 'pages',
    where: { slug: { equals: 'home' } },
    limit: 1,
  })

  if (existingPages.totalDocs === 0) {
    await payload.create({
      collection: 'pages',
      context: { disableRevalidate: true },
      data: {
        title: 'Home',
        slug: 'home',
        hero: {
          type: 'highImpact',
          richText: {
            root: {
              type: 'root',
              children: [
                {
                  type: 'heading',
                  tag: 'h1',
                  children: [{ type: 'text', text: 'Welcome to our website' }],
                  version: 0,
                },
                {
                  type: 'paragraph',
                  children: [
                    {
                      type: 'text',
                      text: 'This is a demo website with multi-language support powered by Payload CMS.',
                    },
                  ],
                  version: 0,
                },
              ],
              direction: 'ltr',
              format: '',
              indent: 0,
              version: 1,
            },
          },
          links: [
            {
              link: { type: 'custom', label: 'Get Started', url: '/posts', appearance: 'default' },
            },
          ],
          media: heroImage,
        },
        layout: [
          {
            blockType: 'content',
            columns: [
              {
                size: 'full',
                richText: {
                  root: {
                    type: 'root',
                    children: [
                      {
                        type: 'heading',
                        tag: 'h2',
                        children: [{ type: 'text', text: 'About this project' }],
                        version: 0,
                      },
                      {
                        type: 'paragraph',
                        children: [
                          {
                            type: 'text',
                            text: 'This project demonstrates internationalization with automatic AI-powered translation. Content created in one language can be translated to all configured locales with a single click.',
                          },
                        ],
                        version: 0,
                      },
                    ],
                    direction: 'ltr',
                    format: '',
                    indent: 0,
                    version: 1,
                  },
                },
              },
            ],
          },
        ],
        meta: {
          title: 'Home | i18n Scanner',
          description:
            'A demo website showcasing multi-language content management with AI-powered translations.',
        },
        _status: 'published',
        publishedAt: new Date().toISOString(),
      },
      locale: 'en',
    })
    payload.logger.info('  ✓ Home page created')
  }

  // Create a sample blog post
  const existingPosts = await payload.find({
    collection: 'posts',
    where: { slug: { equals: 'getting-started' } },
    limit: 1,
  })

  if (existingPosts.totalDocs === 0) {
    await payload.create({
      collection: 'posts',
      context: { disableRevalidate: true },
      data: {
        title: 'Getting Started with i18n',
        slug: 'getting-started',
        content: {
          root: {
            type: 'root',
            children: [
              {
                type: 'heading',
                tag: 'h2',
                children: [{ type: 'text', text: 'Introduction' }],
                version: 0,
              },
              {
                type: 'paragraph',
                children: [
                  {
                    type: 'text',
                    text: 'Internationalization (i18n) is the process of designing and preparing your application to be usable in different languages and regions. This post walks through how our system handles content translation.',
                  },
                ],
                version: 0,
              },
              {
                type: 'heading',
                tag: 'h3',
                children: [{ type: 'text', text: 'How it works' }],
                version: 0,
              },
              {
                type: 'paragraph',
                children: [
                  {
                    type: 'text',
                    text: 'Our system uses a local LLM (via Ollama) to translate content between locales. The translation settings are managed directly from the CMS admin panel, making it easy to configure without code changes.',
                  },
                ],
                version: 0,
              },
            ],
            direction: 'ltr',
            format: '',
            indent: 0,
            version: 1,
          },
        },
        meta: {
          title: 'Getting Started with i18n | Blog',
          description:
            'Learn how internationalization works in our CMS with AI-powered translations.',
        },
        _status: 'published',
        publishedAt: new Date().toISOString(),
      },
      locale: 'en',
    })
    payload.logger.info('  ✓ Blog post created')
  }

  // Seed header navigation
  await payload.updateGlobal({
    slug: 'header',
    context: { disableRevalidate: true },
    data: {
      navItems: [
        { link: { type: 'custom', label: 'Home', url: '/' } },
        { link: { type: 'custom', label: 'Posts', url: '/posts' } },
      ],
    },
  })
  payload.logger.info('  ✓ Header navigation set')

  // Seed footer navigation
  await payload.updateGlobal({
    slug: 'footer',
    context: { disableRevalidate: true },
    data: {
      navItems: [
        { link: { type: 'custom', label: 'Home', url: '/' } },
        { link: { type: 'custom', label: 'Posts', url: '/posts' } },
      ],
    },
  })
  payload.logger.info('  ✓ Footer navigation set')

  // Seed AI Translation settings
  await payload.updateGlobal({
    slug: 'ai-translation',
    data: {
      llmHost: process.env.OLLAMA_HOST || 'http://localhost:11434',
      llmModel: 'llama3.1',
      llmApiKey: 'ollama',
      translationPrompt:
        'Translate the following JSON array from locale "{{localeFrom}}" to locale "{{localeTo}}". Return ONLY a valid JSON array of strings with no markdown, no code fences, no explanation. Input: {{texts}}',
    },
  })
  payload.logger.info('  ✓ AI Translation settings configured')

  payload.logger.info('— Seeding complete!')
}
