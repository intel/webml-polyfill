let nnPolyfill, nnNative;
if (navigator.ml.isPolyfill) {
  nnNative = null;
  nnPolyfill = navigator.ml.getNeuralNetworkContext();
} else {
  nnNative = navigator.ml.getNeuralNetworkContext();
  nnPolyfill = navigator.ml_polyfill.getNeuralNetworkContext();
}

const nativeBackendArray = ['WebML', 'NN', 'BNNS', 'MPS', 'DirectML'];
let acturalNativeBackend = null;

function getOS() {
  var userAgent = window.navigator.userAgent,
      platform = window.navigator.platform,
      macosPlatforms = ['Macintosh', 'MacIntel', 'MacPPC', 'Mac68K'],
      windowsPlatforms = ['Win32', 'Win64', 'Windows', 'WinCE'],
      iosPlatforms = ['iPhone', 'iPad', 'iPod'],
      os = null;

  if (macosPlatforms.indexOf(platform) !== -1) {
    os = 'Mac OS';
  } else if (iosPlatforms.indexOf(platform) !== -1) {
    os = 'iOS';
  } else if (windowsPlatforms.indexOf(platform) !== -1) {
    os = 'Windows';
  } else if (/Android/.test(userAgent)) {
    os = 'Android';
  } else if (!os && /Linux/.test(platform)) {
    os = 'Linux';
  }

  return os;
}

function getNativeAPI() {
  const apiMapping = {
    'Android': 'NN',
    'Windows': 'DirectML',
    'Linux': 'N/A'
  };
  let os = getOS();
  let backend;

  if (os === 'Mac OS') {
    let prefer = getPreferParam();
    if (prefer === 'fast') {
      backend = 'BNNS';
    } else if (prefer === 'sustained') {
      backend = 'MPS';
    }
  } else {
    backend = apiMapping[os];
  }

  return backend;
}

function setActuralNativeAPI(backend) {
  acturalNativeBackend = backend;
}

function getActuralNativeAPI() {
  if (getOS() === 'Mac OS') {
    return acturalNativeBackend;
  } else {
    return getNativeAPI();
  }
}

function getUrlParams( prop ) {
  var params = {};
  var search = decodeURIComponent( window.location.href.slice( window.location.href.indexOf( '?' ) + 1 ) );
  var definitions = search.split( '&' );

  definitions.forEach( function( val, key ) {
    var parts = val.split( '=', 2 );
      params[ parts[ 0 ] ] = parts[ 1 ];
  } );

  return ( prop && prop in params ) ? params[ prop ] : params;
}

function getPreferParam() {
  // workaround for using MPS backend on Mac OS by visiting URL with 'prefer=sustained'
  // workaround for using BNNS backend on Mac OS by visiting URL with 'prefer=fast'
  var prefer = 'sustained';
  var parameterStr = window.location.search.substr(1);
  var reg = new RegExp("(^|&)prefer=([^&]*)(&|$)", "i");
  var r = parameterStr.match(reg);
  if (r != null) {
    prefer = unescape(r[2]);
    if (prefer !== 'fast' && prefer !== 'sustained') {
      console.log("Error: prefer value is invalid, currently it supports 'fast' or 'sustained', switch to use default 'sustained'.");
      prefer = 'sustained';
    }
  }

  return prefer;
}
