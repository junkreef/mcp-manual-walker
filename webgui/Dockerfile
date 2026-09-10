# The test console: nginx, static files, and nothing else.
#
# It is a separate image from the server on purpose. The page is HTML that a
# browser runs, it holds no state and it never touches the databases; giving it
# its own container keeps it off the port the API answers on and lets it be
# left out of a deployment entirely by not starting the service.
#
# There is no build step and no bundler. The console is three files a browser
# understands as they are, so what is in the repository is what runs, and
# editing it needs nothing installed.
FROM nginx:1.29-alpine

# nginx's entrypoint renders every *.template in this directory through
# envsubst before it starts, which is how the upstream can be pointed at
# something other than the compose service without rebuilding the image.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY static/ /usr/share/nginx/html/

# Where /api is forwarded. The default names the compose service and the port
# the server's REST half listens on.
ENV API_UPSTREAM=http://mcp:8001

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["wget", "--quiet", "--tries=1", "--spider", "http://127.0.0.1/"]
