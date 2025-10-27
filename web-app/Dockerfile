FROM node:20-alpine AS deps

ENV NEXT_TELEMETRY_DISABLED=1 \
    npm_config_yes=true

WORKDIR /app

COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm install

FROM node:20-alpine AS builder

ENV NEXT_TELEMETRY_DISABLED=1

WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-alpine AS runner

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /app

RUN addgroup -S nextjs && adduser -S nextjs -G nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package.json ./package.json
COPY --from=deps /app/node_modules ./node_modules

RUN npm prune --production && chown -R nextjs:nextjs /app

USER nextjs

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD wget -qO- http://127.0.0.1:3000/api/healthz > /dev/null 2>&1 || exit 1

CMD ["npm", "start"]
