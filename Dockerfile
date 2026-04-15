##############################################
# Unified Multi-Stage Dockerfile
# Targets:
#   deps          -> pnpm install (single run)
#   build-app     -> Next.js build
#   runtime-app   -> Production Next.js runtime (standalone)
# Usage examples:
#   docker build --target deps -t i18n-deps-local .
#   docker build --target runtime-app -t i18n-app-local .
##############################################

FROM node:20.20.2-alpine AS base
RUN npm config set registry https://hub.talkdeskapp.com:8443/repository/talkdesk-npm/ 
RUN npm install -g pnpm@10.33.0

##############################################
# deps stage: install all dependencies once
##############################################
FROM base AS deps
WORKDIR /app
RUN apk add --no-cache libc6-compat
# Copy only dependency graph relevant files (changes invalidate cache intentionally)
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY tsconfig.json next.config.ts ./
# Source required for build
COPY public ./public
COPY src ./src
# --ignore-scripts skips lifecycle hooks (husky install, git config) that need git
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm install --frozen-lockfile --ignore-scripts

##############################################
# build-app stage: build Next.js
##############################################
FROM deps AS build-app
RUN pnpm run build

##############################################
# runtime-app stage: minimal production image
##############################################
FROM base AS runtime-app
WORKDIR /app
ENV NODE_ENV=production
# Create dedicated non-root user
RUN addgroup -g 20000 tdgroup && adduser -S -u 20000 -G tdgroup tduser
# Copy only the standalone build output
# Note: turbopack.root in next.config.ts nests standalone output under app/
COPY --from=build-app /app/public ./public
COPY --from=build-app /app/.next/standalone/app ./
COPY --from=build-app /app/.next/static ./.next/static
# Ensure ownership and writable cache/tmp dirs
RUN chown -R tduser:tdgroup /app /tmp && chmod 1777 /tmp
USER tduser
EXPOSE 3000
ENV PORT=3000
CMD ["node", "server.js"]
