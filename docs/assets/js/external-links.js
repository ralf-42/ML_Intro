// External Links - Open in new tab
(function() {
  console.log('[EXT] Script file loaded');

  function process() {
    console.log('[EXT] Processing...');
    var links = document.querySelectorAll('a');
    var count = 0;

    for (var i = 0; i < links.length; i++) {
      var link = links[i];
      var href = link.href;

      if (href && href.indexOf('http') === 0 && href.indexOf(window.location.hostname) === -1) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
        count++;
      }
    }

    console.log('[EXT] Processed: ' + count);
  }

  if (document.readyState === 'complete') {
    process();
  } else {
    window.addEventListener('load', process);
  }
})();
