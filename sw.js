const CACHE_NAME = 'v61';
let urlsToCache = [];

self.addEventListener('install', function (event) {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function (cache) {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', function (event) {
  // check if request is made by chrome extensions or web page
  // if request is made for web page url must contains http.
  if (!(event.request.url.indexOf('http') === 0)) return; // skip the request. if request is not made with http protocol

  event.respondWith(
    caches.match(event.request)
      .then(function (response) {
        if (response) {
          return response;
        }

        let fetchRequest = event.request.clone();

        return fetch(fetchRequest).then(
          function (response) {
            if (!response || response.status !== 200 ||
              (response.type !== 'basic' && response.type !== 'cors')) {
              return response;
            }

            let responseToCache = response.clone();

            caches.open(CACHE_NAME).then(function (cache) {
              cache.put(event.request, responseToCache);
            });

            return response;
          }
        );
      })
  );
});

self.addEventListener('activate', function (event) {
  let cacheWhitelist = [CACHE_NAME];

  event.waitUntil(
    caches.keys().then(function (keyList) {
      return Promise.all(keyList.map(function (key) {
        if (cacheWhitelist.indexOf(key) === -1) {
          return caches.delete(key);
        }
      }));
    })
  );
});
