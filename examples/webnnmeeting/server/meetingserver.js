/*
 * Copyright © 2018 Intel Corporation. All Rights Reserved.
 */
'use strict'

const fs = require('fs')
const path = require('path')
const https = require('https')
const express = require('express')
const cors = require('cors')
const bodyParser = require('body-parser')
const app = express()

const config = require('../config')
const rest = require('./authrequest')

const corsOptions = {
  origin: '*',
  optionsSuccessStatus: 200,
  methods: ['GET', 'POST', 'PUT', 'PATCH', 'OPTIONS', 'DELETE']
  // alloweHeaders:['Content-Type','Authorization']
}

// Directory 'public' for static files
// app.use(express.static(__dirname + '/html'))
app.use(bodyParser.json())
app.use(
  bodyParser.urlencoded({
    extended: true
  })
)

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  next()
})

// Set rejectUnauthorized to true for production use
const request = rest(
  config.webrtcserver.id,
  config.webrtcserver.key,
  config.webrtcserver.url + ':' + config.webrtcserver.port,
  false
)

// Prepare sample room before start-up
const prepareWebNNRoom = new Promise((resolve, reject) => {
  const checkresponse = (resp) => {
    const rooms = JSON.parse(resp)
    let webnnroomid = null
    // Find sample room
    for (const room of rooms) {
      if (room.name === 'sampleRoom') {
        webnnroomid = room._id
        break
      }
    }
    if (webnnroomid) {
      resolve(webnnroomid)
    } else {
      // Try create
      const createbody = JSON.stringify({
        name: 'sampleRoom',
        options: {}
      })
      const createok = (resp) => {
        resolve(JSON.parse(resp)._id)
      }
      request('POST', '/v1/rooms', createbody, createok, reject)
    }
  }

  request('GET', '/v1/rooms?page=1&per_page=100', null, checkresponse, reject)
})

function onRequestFail(err) {
  console.log('Request Fail:', err)
}

prepareWebNNRoom
  .then((webnnroom) => {
    console.log('sampleRoom Id:', webnnroom)

    // Create token API with default room
    app.post('/createToken/', cors(corsOptions), function(req, res) {
      const tokenroom = req.body.room || webnnroom
      request('POST', '/v1/rooms/' + tokenroom + '/tokens', req.body)
        .then((imRes) => {
          res.writeHead(imRes.statusCode, imRes.headers)
          imRes.pipe(res)
        })
        .catch(onRequestFail)
    })

    // Route internal REST interface
    app.use(cors(corsOptions), function(req, res) {
      request(req.method, '/v1' + req.path, req.body)
        .then((imRes) => {
          res.writeHead(imRes.statusCode, imRes.headers)
          imRes.pipe(res)
        })
        .catch(onRequestFail)
    })

    // Start HTTP server
    app.listen(config.restapiserver.httpport)

    // Start HTTPS server
    try {
      https
        .createServer(
          {
            cert: fs.readFileSync(path.resolve(config.certificate.cert)),
            key: fs.readFileSync(path.resolve(config.certificate.key))
          },
          app
        )
        .listen(config.restapiserver.httpsport)
      console.log(
        'WebNN Meeting Rest API Server HTTPS Port: ' +
          config.restapiserver.httpsport
      )
      // console.log(
      //   'WebNN Meeting Rest API Server HTTP Port: ' +
      //     config.restapiserver.httpport
      // )
    } catch (e) {
      console.log(e)
    }
  })
  .catch((e) => {
    console.log('Failed to intialize webnnroom', e)
  })
