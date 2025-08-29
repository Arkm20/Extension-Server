const CACHE_NAME = 'animex-v2';
const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/library.html',
  '/offline.html',
  // Essential resources
  '/Resources/manifest.json',
  '/Resources/Images/icon-196.png',
  // External libs & fonts
  'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap',
  'https://fonts.googleapis.com/css2?family=Bitcount+Grid+Single:wght@100..900&display=swap',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.11.338/pdf.worker.min.js',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/all.css',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/sharp-solid.css',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/sharp-regular.css',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/sharp-light.css',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/duotone.css',
  'https://site-assets.fontawesome.com/releases/v6.7.2/css/brands.css',
];

// Install: cache critical assets
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        return Promise.allSettled(
          PRECACHE_URLS.map(url =>
            cache.add(url).catch(err => {
              console.warn('SW cache add failed:', url, err);
              return null; // Don't fail the entire install
            })
          )
        );
      })
      .then(() => self.skipWaiting())
  );
});

// Activate: clean up old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(key => key !== CACHE_NAME)
          .map(key => caches.delete(key))
      )
    ).then(() => self.clients.claim())
  );
});

// Fetch: Cache first for app resources, network first for external APIs
self.addEventListener('fetch', event => {
  if (event.request.method !== 'GET') return;

  const url = new URL(event.request.url);

  // Don't intercept special protocols
  if (['blob:', 'data:', 'chrome-extension:', 'moz-extension:'].includes(url.protocol)) {
    return;
  }

  // For navigation requests, use a network-first strategy.
  // This ensures users get the latest version if online,
  // but falls back to the cache and then the offline page.
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then(response => {
          // If we get a good response, cache it for next time.
          if (response && response.status === 200) {
            const responseToCache = response.clone();
            caches.open(CACHE_NAME).then(cache => {
              cache.put(event.request, responseToCache);
            });
          }
          return response;
        })
        .catch(() => {
          // Network failed, try to serve from cache.
          return caches.match(event.request).then(cachedResponse => {
            // If in cache, serve it. Otherwise, show the offline page.
            return cachedResponse || caches.match('/offline.html');
          });
        })
    );
    return;
  }

  // For all other requests (CSS, JS, images), use a cache-first strategy.
  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      // Return from cache if found.
      if (cachedResponse) {
        return cachedResponse;
      }
      // Not in cache, so try the network.
      return fetch(event.request).then(networkResponse => {
        // If we get a good response, cache it.
        if (networkResponse && networkResponse.status === 200) {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then(cache => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      });
      // If fetch fails, the browser will handle the error (e.g., broken image).
      // No explicit .catch is needed here unless we want to provide specific fallbacks for assets.
    })
  );
});

self.addEventListener('message', event => {
  if (event.data && event.data.action) {
    switch (event.data.action) {
      case 'clear_page_cache':
        console.log('SW: Clearing page cache...');
        // URLs to clear: HTML pages and external CDN resources
        const urlsToClear = PRECACHE_URLS.filter(url =>
          url.endsWith('.html') || url.includes('https://') || url === '/'
        );
        console.log('SW: URLs to clear:', urlsToClear);

        event.waitUntil(
          caches.open(CACHE_NAME).then(cache => {
            const promises = urlsToClear.map(url => cache.delete(url));
            return Promise.all(promises).then(() => {
              console.log('SW: Page cache cleared.');
              // Notify clients
              self.clients.matchAll({ includeUncontrolled: true, type: 'window' }).then(clients => {
                clients.forEach(client => client.postMessage({ action: 'cacheCleared', type: 'page' }));
              });
            });
          })
        );
        break;

      case 'clear_all_caches':
        console.log('SW: Clearing all caches...');
        event.waitUntil(
          caches.keys().then(keys =>
            Promise.all(
              keys.map(key => caches.delete(key))
            )
          ).then(() => {
            console.log('SW: All caches deleted.');
            self.clients.matchAll({ includeUncontrolled: true, type: 'window' }).then(clients => {
                clients.forEach(client => client.postMessage({ action: 'cacheCleared', type: 'all' }));
            });
          })
        );
        break;
    }
  }
});
