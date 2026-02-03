FROM node:20-alpine AS build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json* ./
ENV NODE_ENV=development
RUN npm install --include=dev
COPY frontend ./
ARG VITE_API_BASE=/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN node /web/node_modules/vite/bin/vite.js build

FROM nginx:1.25-alpine
COPY nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /web/dist /usr/share/nginx/html
