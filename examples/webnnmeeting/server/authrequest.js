// Copyright (C) <2018> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

'use strict'

const crypto = require('crypto')
const http = require('http')
const https = require('https')

function signWithKey(data, key) {
  const hex = crypto
    .createHmac('sha256', key)
    .update(data)
    .digest('hex')
  return Buffer.from(hex).toString('base64')
}

// Send REST request to owt-server with auth
module.exports = function requestor(service, key, url, rejectUnauthorized) {
  const URL = require('url').URL
  const parsedUrl = new URL(url)
  const config = {
    service,
    key,
    host: parsedUrl.hostname,
    port: parsedUrl.port || 80,
    secure: parsedUrl.protocol === 'https:',
    rejectUnauthorized
  }

  const request = function(method, resource, body, onOk, onError) {
    const timestamp = new Date().getTime()
    const cnounce = crypto.randomBytes(8).toString('hex')

    const toSign = timestamp + ',' + cnounce
    const signed = signWithKey(toSign, config.key)

    let header =
      'MAuth realm=http://marte3.dit.upm.es,mauth_signature_method=HMAC_SHA256'
    header += ',mauth_serviceid='
    header += config.service
    header += ',mauth_cnonce='
    header += cnounce
    header += ',mauth_timestamp='
    header += timestamp
    header += ',mauth_signature='
    header += signed

    const options = {
      hostname: config.host,
      port: config.port,
      path: resource,
      method, // 'POST',
      headers: {
        Authorization: header,
        'Content-Type': 'application/json'
      },
      rejectUnauthorized: config.rejectUnauthorized
    }
    if (body) {
      if (typeof body === 'object') {
        body = JSON.stringify(body)
      }
      options.headers['Content-Length'] = Buffer.byteLength(body)
    }

    const httpHttps = config.secure ? https : http

    const responsePromise = new Promise((resolve, reject) => {
      const req = httpHttps.request(options, (res) => {
        // Keep res.statusCode, res.headers
        resolve(res)
      })
      req.on('error', (e) => {
        reject(e)
      })
      // Write data to request body
      if (body) {
        req.write(body)
      }
      req.end()
    })

    if (typeof onOk === 'function' && typeof onError === 'function') {
      responsePromise
        .then((res) => {
          res.setEncoding('utf8')
          let data = ''
          res.on('data', (chunk) => {
            data += chunk
          })
          res.on('end', () => {
            const successCode = [100, 200, 201, 202, 203, 204, 205]
            if (successCode.includes(res.statusCode)) {
              onOk(data)
            } else {
              onError(data)
            }
          })
        })
        .catch(onError)
    } else {
      // Return promise
      return responsePromise
    }
  }

  return request
}
