// Copyright (C) <2018> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

// REST samples. It sends HTTP requests to sample server, and sample server sends requests to conference server.
// Both this file and sample server are samples.
'use strict'

const config = require('../../config.js')

const send = function(method, path, body, onRes, host) {
  const req = new XMLHttpRequest()
  req.onreadystatechange = function() {
    if (req.readyState === 4) {
      onRes(req.responseText)
    }
  }
  const url = generateUrl(host, path)
  req.open(method, url, true)
  req.setRequestHeader('Content-Type', 'application/json')
  if (body !== undefined) {
    req.send(JSON.stringify(body))
  } else {
    req.send()
  }
}

const generateUrl = function(host, path) {
  let url
  if (host !== undefined) {
    url = host + path // Use the host user set.
    url = url.replace(
      config.nuxtserver.httpsport,
      config.restapiserver.httpsport
    )
  } else {
    const u = new URL(document.URL)
    url = u.origin + path // Get the string before last '/'.
    url = url.replace(
      config.nuxtserver.httpsport,
      config.restapiserver.httpsport
    )
  }
  return url
}

const onResponse = function(result) {
  if (result) {
    try {
      console.info('Result:', JSON.parse(result))
    } catch (e) {
      console.info('Result:', result)
    }
  } else {
    console.info('Null')
  }
}

const mixStream = function(room, stream, view, host) {
  const jsonPatch = [
    {
      op: 'add',
      path: '/info/inViews',
      value: view
    }
  ]
  send(
    'PATCH',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    onResponse,
    host
  )
}

const echoCancellation = function(room, stream, track, status, callback, host) {
  const jsonPatch = []
  if (track === 'audio' || track === 'av') {
    jsonPatch.push({
      op: 'replace',
      path: '/media/audio/format/echoCancellation',
      value: status
    })
  }
  send(
    'PATCH',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    callback,
    host
  )
}

// https://software.intel.com/sites/products/documentation/webrtc/restapi/
const updateStream = function(room, stream, host) {
  const jsonPatch = [
    { op: 'replace', path: '/media/video/status', value: 'inactive' }
  ]
  send(
    'PATCH',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    onResponse,
    host
  )
}

const deleteStream = function(room, stream, host) {
  const jsonPatch = [{ name: 'deleteStream' }]
  send(
    'DELETE',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    onResponse,
    host
  )
}

const startStreamingIn = function(room, inUrl, host) {
  const options = {
    url: inUrl,
    media: {
      audio: 'auto',
      video: true
    },
    transport: {
      protocol: 'udp',
      bufferSize: 2048
    }
  }
  send('POST', '/rooms/' + room + '/streaming-ins', options, onResponse, host)
}

const createToken = function(room, user, role, callback, host) {
  const body = {
    preference: { isp: 'isp', region: 'region' },
    user,
    role
  }
  if (room) {
    send('POST', '/rooms/' + room + '/tokens/', body, callback, host)
  } else {
    send('POST', '/createToken/', body, callback, host)
  }
}

const getStreams = function(room, callback, host) {
  const resCb = function(result) {
    if (result) {
      try {
        callback(JSON.parse(result))
      } catch (e) {
        callback(null)
      }
    } else {
      callback(null)
    }
  }
  send('GET', '/rooms/' + room + '/streams/', undefined, resCb, host)
}

const pauseStream = function(room, stream, track, callback, host) {
  const jsonPatch = []
  if (track === 'audio' || track === 'av') {
    jsonPatch.push({
      op: 'replace',
      path: '/media/audio/status',
      value: 'inactive'
    })
  }

  if (track === 'video' || track === 'av') {
    jsonPatch.push({
      op: 'replace',
      path: '/media/video/status',
      value: 'inactive'
    })
  }
  send(
    'PATCH',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    callback,
    host
  )
}

const playStream = function(room, stream, track, callback, host) {
  const jsonPatch = []
  if (track === 'audio' || track === 'av') {
    jsonPatch.push({
      op: 'replace',
      path: '/media/audio/status',
      value: 'active'
    })
  }

  if (track === 'video' || track === 'av') {
    jsonPatch.push({
      op: 'replace',
      path: '/media/video/status',
      value: 'active'
    })
  }
  send(
    'PATCH',
    '/rooms/' + room + '/streams/' + stream,
    jsonPatch,
    callback,
    host
  )
}

export {
  send,
  generateUrl,
  onResponse,
  mixStream,
  updateStream,
  deleteStream,
  echoCancellation,
  startStreamingIn,
  createToken,
  getStreams,
  pauseStream,
  playStream
}
