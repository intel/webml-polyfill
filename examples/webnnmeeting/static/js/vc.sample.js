// License
var localStream = null;
var localScreen = null;
var localName = 'Anonymous';
var localId = null;
var localScreenId = null;
var users = [];
var progressTimeOut = null;
var smallRadius = 60;
var largeRadius = 120;
var isMouseDown = false;
var mouseX = null;
var mouseY = null;
var MODES = {
  GALAXY: 'galaxy',
  MONITOR: 'monitor',
  LECTURE: 'lecture'
};
var mode = MODES.GALAXY;
var SUBSCRIBETYPES = {
  FORWARD: 'forward',
  MIX: 'mix'
};
var subscribeType = SUBSCRIBETYPES.FORWARD;
var isScreenSharing = false;
var isLocalScreenSharing = false;
var remoteScreen = null;
var remoteScreenName = null;
var isMobile = false;
var streamObj = {};
var streamIndices = {};
var hasMixed = false;
var isSmall = false;
var isPauseAudio = true;
var isPauseVideo = false;
var isOriginal = true;
var isAudioOnly = false;

var showInfo = null;
var showLevel = null;
var scaleLevel = 3/4;
var refreshMute = null;

var currentRegions = null;

var localPublication = null;
var localScreenPubliction = null;
var joinResponse = null;

var localResolution = null;
var remoteMixedSub = null;
var subList = {};
var screenSub = null;

var room = null;
var roomId = null;

const remoteStreamMap = new Map();
const forwardStreamMap = new Map();

function login() {
  const value = $('#login-input').val();
  if (value !== '') {
    localName = $('<textarea/>').text(value).html();
    $('#login-panel').addClass('pulse');
    $('#login-panel').hide();
    $('#container').show();
    if(navigator.webkitGetUserMedia){
      $('#codec').parent().css('display', 'block');
      $('#bandwidth').parent().css('display', 'block');
    }
    initConference();
  }
  if (isMobile && typeof document.body.webkitRequestFullScreen === 'function') {
    document.body.webkitRequestFullScreen();
  }
}

function alertCert(signalingHost) {
  const $d = $('#m-dialog');
  $d.empty();
  const infoText = 'The security certificate of the following url ' +
    'is not trusted by your computer\'s operating system. ' +
    'If you confirm to continue, click the url and proceed to the unsafe host, then come back' +
    'to this page and refresh.';
  const info = $('<div/>', {
    text: infoText
  });
  const anchor = $('<a/>', {
      text: `${signalingHost}/socket.io/`,
      target: '_blank',
      href: `${signalingHost}/socket.io/`
  });
  info.appendTo($d);
  anchor.appendTo($d);
  $d.show();
  $d.dialog();
}

function toggleLoginSetting() {
  $('#default-login').slideToggle();
  $('#setting-login').slideToggle();
}

function loginDisableVideo() {
  $('#login-resolution').slideUp();
}

function loginEnableVideo() {
  $('#login-resolution').slideDown();
}

function exit() {
  if (confirm('Are you sure to exit?')) {
    // window.open('', '_self').close();
    userExit();
  }
}

function userExit() {
  if (localStream) {
    localStream.mediaStream.getTracks().forEach(track => {
      track.stop();
    });
  }
  if (localScreen) {
    localScreen.mediaStream.getTracks().forEach(track => {
      track.stop();
    });
    $('#screen-btn').removeClass('disabled');
  }
  room.leave();

  users = [];
  subList = {};
  streamObj = {};
  streamIndices = {};
  $("#video-panel div[id^=client-]").remove();
  $("#localScreen").remove();
  $("#screen").remove();
  $("#container").hide();
  $("#login-panel").removeClass("pulse").show();
  $("#user-list").empty();
  $('#video-panel').empty();
  localStream = undefined;
  isAudioOnly = false;
  clearInterval(refreshMute);
}

function processRemoteStream(stream) {
  remoteStreamMap.set(stream.id, stream);
  let enedListener = (event) => {
    console.log(`remote stream ${stream.id} is ended`);
    remoteStreamMap.delete(stream.id);
    // destroyStreamUi(stream);
  };
  let activeaudioinputchangeListener = (event) => {
    console.log('activeaudioinputchange event triggered: ', event);
  };
  let layoutchangeListener = (event) => {
    console.log('layoutchange event triggered: ', event);
  };
  let update = (event) => {
    console.log('streamupdate event triggered: ', event, stream.id);
  }
  stream.addEventListener('ended', enedListener);
  stream.addEventListener('activeaudioinputchange', activeaudioinputchangeListener);
  stream.addEventListener('layoutchange', layoutchangeListener);
  stream.addEventListener('update', update);
}

function subscribeStream(stream) {
  console.info('subscribing:', stream.id);
  var videoOption = !isAudioOnly;
  room.subscribe(stream, {video: videoOption}).then(subscription => {
    console.info('subscribed: ',subscription.id);
    addVideo(stream, false);
    subList[subscription.id] = subscription;
    console.info("add success");
    streamObj[stream.id] = stream;
    if(stream.source.video === 'mixed'){
      remoteMixedSub = subscription;
    }
    if(stream.source.video === 'screen-cast'){
      screenSub = subscription;
      stream.addEventListener('ended', function(event) {
        changeMode(MODES.LECTURE);
        setTimeout(function() {
          $('#local-screen').remove();
          $('#screen').remove();
          shareScreenChanged(false, false);
          if (subscribeType === SUBSCRIBETYPES.MIX) {
            changeMode(mode, $("div[isMix=true]"));
          } else {
            changeMode(mode);
          }
        }, 800);
      });
    }
    setTimeout(function() {
      subscription.getStats().then( report => {
        console.info(report);
        report.forEach(function(item,index){
          if(item.type === 'ssrc' && item.mediaType === 'video'){
            scaleLevel = parseInt(item.googFrameHeightReceived)/parseInt(item.googFrameWidthReceived);
            console.info(scaleLevel);
          }
        });
        resizeStream(mode);
      }, err => {
        console.error('stats error: ' + err);
      });
    }, 1000);
    // monitor(subscription);
  }, err => {
    console.error('subscribe error: ' + err);
  });
}

function initConference() {
  if ($('#subscribe-type').val() === 'mixed') {
    subscribeType = SUBSCRIBETYPES.MIX;
    mode = MODES.LECTURE;
    $("#monitor-btn").addClass("disabled");
    $("#galaxy-btn").addClass("disabled");
  } else {
    subscribeType = SUBSCRIBETYPES.FORWARD;
    mode = MODES.GALAXY;
    $("#monitor-btn").removeClass("disabled");
    $("#galaxy-btn").removeClass("disabled");
  }

  $('#userNameDisplay').html("Logged in as: " + localName);

  var bandWidth = 100,
  localResolution = new Owt.Base.Resolution(320,240);
  if ($('#login-480').hasClass('selected')) {
    bandWidth = 500;
    localResolution = new Owt.Base.Resolution(640,480);
  } else if ($('#login-720').hasClass('selected')) {
    bandWidth = 1000;
    localResolution = new Owt.Base.Resolution(1280,720);
  }

  var avTrackConstraint = {};
  if ($("#login-audio-video").hasClass("selected")) {
    //TODO: maybe the room constraint,need to test a new room for more information
    avTrackConstraint = {
      audio: {
        source: "mic",
      },
      video:{
        resolution: localResolution,
        frameRate:24,
        source:'camera'
      },
    }
    console.log(avTrackConstraint);
  } else {
    avTrackConstraint = {
      audio: {
        source: "mic",
      },
      video: false,
    };
    isAudioOnly = true;
  }

  function createLocal() {
    let mediaStream;
    Owt.Base.MediaStreamFactory.createMediaStream(avTrackConstraint).then(stream => {
        mediaStream = stream;
        console.info('Success to create MediaStream');
        localStream = new Owt.Base.LocalStream(
          mediaStream,new Owt.Base.StreamSourceInfo(
            'mic', 'camera')
        );
        console.log('local stream:', localStream);
        localId = localStream.id;
        addVideo(localStream, true);
        $('#text-send,#send-btn').show();
        room.publish(localStream).then(publication => {
          localPublication = publication;
          isPauseAudio = false;
          toggleAudio();
          isPauseVideo = true;
          toggleVideo();
          mixStream(roomId,localPublication.id,'common');
          console.info('publish success');
          streamObj[localStream.id] = localStream;
            publication.addEventListener('error', (err) => {
            console.log('Publication error: ' + err.error.message);
          });
        }, err => {
          console.log('Publish error: ' + err);
        });
      }, err => {
        console.error('Failed to create MediaStream, ' + err);
        if (err.name === "OverconstrainedError") {
          if(confirm("your camrea can't support the resolution constraints, please leave room and select a lower resolution")) {
            userExit();
          }
        }
      });
  }

  createToken(roomId, localName, 'presenter', function(response) {
    var token = response;
    if (!room) {
      room = new Owt.Conference.ConferenceClient();
      addRoomEventListener();
    }
    room.join(token).then(resp => {
        roomId = resp.id;
        var getLoginUsers = resp.participants;
        var streams = resp.remoteStreams;
        console.log(resp);
        getLoginUsers.map(function(participant){
          participant.addEventListener('left', () => {
            //TODO:send message for notice everyone the participant has left maybe no need
            deleteUser(participant.id);
          });
          users.push({
            id: participant.id,
            userId: participant.userId,
            role: participant.role
          });
        });
        loadUserList();
        createLocal();
        streamObj = {};

        for (const stream of streams){
          if (stream.source.audio === 'mixed' && stream.source.video === 'mixed') {
            console.log("Mix stream id: " + stream.id);
            stream.addEventListener('layoutChanged', function(regions) {
              console.info('stream', stream.id, 'VideoLayoutChanged');
              currentRegions = regions;
            });
          }
          console.info('stream in conference:', stream.id);
          streamObj[stream.id] = stream;

          let isMixStream = (stream.source.audio === 'mixed');
          if ((subscribeType === SUBSCRIBETYPES.FORWARD && !isMixStream) ||
              (subscribeType === SUBSCRIBETYPES.MIX && isMixStream) ||
              (stream.source.video === 'screen-cast')) {
            subscribeStream(stream); 
          }
        }

        refreshMuteState();
    }, err => {
      console.log("server connect failed: " + err);
      if (err.message.indexOf('connect_error:') >= 0) {
        const signalingHost = err.message.replace('connect_error:', '');
        alertCert(signalingHost);
      }
    });
  });
}

function refreshMuteState() {
  refreshMute = setInterval(() => {
    getStreams(roomId, (streams) => {
      forwardStreamMap.clear();
      for (const stream of streams) {
        // console.log(stream);
        if (stream.type === 'forward') {
          forwardStreamMap.set(stream.id, stream);
          if (stream.media.audio) {
            const clientId = stream.info.owner;
            const muted = (stream.media.audio.status === 'inactive');
            chgMutePic(clientId, muted);
          }
        }
      }
    });
  }, 1000);
}

function monitor(subscription){
  var bad = 0, lost = 0, rcvd = 0, lostRate = 0, bits = 0, bitRate = 0;
  var current = 0, level = 0;
  showInfo = setInterval(function() {
    subscription.getStats().then((report) => {
      var packetsLost = 0;
      var packetsRcvd = 0;
      var bitsRcvd =0;
      report.forEach(function(value, key){
        if(key.indexOf('RTCInboundRTPVideoStream') != -1) {
          packetsLost = parseInt(value.packetsLost);
          packetsRcvd = parseInt(value.packetsReceived);
          bitsRcvd = parseInt(value.bytesReceived);
          bitRate = ((bitsRcvd-bits)*8/1000/1000).toFixed(3);
          lostRate = (packetsLost-lost) / (packetsRcvd-rcvd);


          $('#rcvd').html(packetsRcvd);
          $('#lost').html(packetsLost);
          $('#bitRate').html(bitRate);
          $('#codec').html(report.get(value.codecId).mimeType.split('/')[1]);
          
          lost = packetsLost;
          rcvd = packetsRcvd;
          bits = bitsRcvd;

          var lostRateNum = parseInt(lostRate*100);
          if(lostRateNum <2) {
            level = 4;
          } else if(lostRateNum <5) {
            level = 3;
          } else if(lostRateNum <10) {
            level = 2;
          } else {
            level = 1;
          }
        }
        if(key.indexOf('RTCIceCandidatePair') != -1) {
          $('#bandwidth').html((parseInt(value.availableIncomingBitrate)/1024/1024).toFixed(3));
        }
      });
    }, err => {
      console.error(err);
    })
  },1000);
  showLevel = setInterval(function() {
    level = level > 4 ? 4 : level;
    if (current < level) {
      current++;
      $('#wifi'+current).css('display', 'block').siblings().css('display', 'none');
    } else if (current > level) {
      current--;
      $('#wifi'+current).css('display', 'block').siblings().css('display', 'none');
    }
    if ((level < 2) && !isPauseVideo) {
      if(bad < 8){
        bad++;
      }
    } else if (bad > 0) {
        bad--;
    }
    if (bad >= 8 && $('#promt').css('opacity') == '0') {
      $('#promt').css('opacity', '1');
    } else if(bad == 0 && $('#promt').css('opacity') == '1') {
      $('#promt').css('opacity', '0');
    }
  }, 1000);
}

function stopMonitor() {
  clearInterval(showInfo);
  clearInterval(showLevel);
}

function loadUserList() {
  for (var u in users) {
    addUserListItem(users[u], true);
  }
}

function addUserListItem(user, muted) {
  var muteBtn =
    '<img src="img/mute_white.png" class="muteShow" isMuted="true"/>';
  var unmuteBtn =
    '<img src="img/unmute_white.png" class="muteShow" isMuted="false"/>';
  var muteStatus = muted ? muteBtn : unmuteBtn;
  $('#user-list').append('<li><div class="userID">' + user.id +
    '</div><img src="img/avatar.png" class="picture"/><div class="name">' +
    user.userId + '</div>' + muteStatus + '</li>');
}

function chgMutePic(id, muted) {
  var line = $('li:contains(' + id + ')').children('.muteShow');
  if (muted) {
    line.attr('src', "img/mute_white.png");
    line.attr('isMuted', true);
  } else {
    line.attr('src', "img/unmute_white.png");
    line.attr('isMuted', false);
  }
}

function addRoomEventListener() {
  room.addEventListener('streamadded', (streamEvent) => {
    console.log('streamadded', streamEvent);
    var stream = streamEvent.stream;

    if (localStream && localStream.id === stream.id) {
      return;
    }
    if (stream.source.audio === 'mixed' && stream.source.video === 'mixed') {
      if (subscribeType !== SUBSCRIBETYPES.MIX) {
        return;
      }
      // subscribe mix stream
      thatName = "MIX Stream";
    } else {
      if (stream.source.video === 'screen-cast') {
        thatName = "Screen Sharing";
        if (isLocalScreenSharing) {
          return;
        }
      } else if (subscribeType !== SUBSCRIBETYPES.FORWARD) {
        return;
      }
    }

    var thatId = stream.id;
    if (stream.source.audio === 'mixed' && stream.source.video === 'mixed') {
      thatName = "MIX Stream";
    } else if ( stream.source.video === 'screen-cast') {
      thatName = "Screen Sharing";
    }

    // add video of non-local streams
    if (localId !== thatId && localScreenId !== thatId && localName !== getUserFromId(stream.origin).userId) {
      subscribeStream(stream);
    }
  });

  room.addEventListener('participantjoined', (event) => {
    console.log('participantjoined', event);
    if(event.participant.userId !== 'user' && getUserFromId(event.participant.id) === null){
      //new user
      users.push({
        id: event.participant.id,
        userId: event.participant.userId,
        role: event.participant.role
      });
      event.participant.addEventListener('left', () => {
        if(event.participant.id !== null && event.participant.userId !== undefined){
          sendIm(event.participant.userId + ' has left the room ', 'System');
          deleteUser(event.participant.id);
        } else {
          sendIm('Anonymous has left the room.', 'System');
        }
      });
      console.log("join user: " + event.participant.userId);
      addUserListItem(event.participant,true);
      //no need: send message to all for initId
    }
  });

  room.addEventListener('messagereceived', (event) => {
    console.log('messagereceived', event);
    var user = getUserFromId(event.origin);
    if (!user) return;
    var receivedMsg = JSON.parse(event.message);
    if(receivedMsg.type == 'msg'){
      if(receivedMsg.data != undefined) {
        var time = new Date();
        var hour = time.getHours();
        hour = hour > 9 ? hour.toString() : '0' + hour.toString();
        var mini = time.getMinutes();
        mini = mini > 9 ? mini.toString() : '0' + mini.toString();
        var sec = time.getSeconds();
        sec = sec > 9 ? sec.toString() : '0' + sec.toString();
        var timeStr = hour + ':' + mini + ':' + sec;
        var color = getColor(user.userId);
        $('<p class="' + color + '">').html(timeStr + ' ' + user.userId +'<br />')
        .append(document.createTextNode(receivedMsg.data)).appendTo('#text-content');
        $('#text-content').scrollTop($('#text-content').prop('scrollHeight'));
      }
    }
  }); 
}

function shareScreen() {
  if ($('#screen-btn').hasClass('disabled')) {
    return;
  }
  sendIm('You are sharing screen now.');
  $('#video-panel .largest').removeClass("largest");
  $('#video-panel').append(
    '<div id="local-screen" class="client clt-0 largest"' +
    '>Screen Sharing</div>').addClass('screen');
  changeMode(MODES.LECTURE, $('#local-screen'));
  var width = screen.width,
    height = screen.height;

  var screenSharingConfig = {
    audio: {
      source: "screen-cast"
    },
    video:{
      resolution:{
        "width": width,
        "height":height
      },
      frameRate:20,
      source:'screen-cast'
    },
    //extensionId:'pndohhifhheefbpeljcmnhnkphepimhe'
  }
  Owt.Base.MediaStreamFactory.createMediaStream(screenSharingConfig).then( stream => {
    localScreen = new Owt.Base.LocalStream(stream,new Owt.Base.StreamSourceInfo('screen-cast','screen-cast'));
    console.info(localScreen);
    localScreenId = localScreen.id;
    var screenVideoTracks = localScreen.mediaStream.getVideoTracks();
    for (const screenVideoTrack of screenVideoTracks) {
      screenVideoTrack.addEventListener('ended', function(e) {
        changeMode(MODES.LECTURE);
        console.log('unpublish');
        setTimeout(function() {
          $('#local-screen').remove();
          $('#screen').remove();
          shareScreenChanged(false, false);
          if (subscribeType === SUBSCRIBETYPES.MIX) {
            changeMode(mode, $("div[isMix=true]"));
          } else {
            changeMode(mode);
          }
        }, 800);
        localScreenPubliction.stop();
      });
    }
    changeMode(MODES.LECTURE,$('#local-screen'));
    room.publish(localScreen).then( publication => {
      console.info('publish success');
      localScreenPubliction = publication;
    }, err => {
      console.error('localsreen publish failed');
    });
  }, err => {
    console.error('create localscreen failed');
    changeMode(MODES.LECTURE);
    $('#local-screen').remove();
    $('#screen').remove();
    shareScreenChanged(false, false);
    //TODO: limit to https
    if (subscribeType === SUBSCRIBETYPES.MIX) {
      changeMode(mode, $("div[isMix=true]"));
    }
    // if (window.location.protocol === "https:" && subscribeType ===
    //   SUBSCRIBETYPES.MIX) {
    //   changeMode(mode, $("div[isMix=true]"));
    // }
  });

  shareScreenChanged(true, true);
}

// update screen btn when sharing
function shareScreenChanged(ifToShare, ifToLocalShare) {
  isScreenSharing = ifToShare;
  isLocalScreenSharing = ifToLocalShare;
  $('#screen-btn').removeClass('disabled selected');
  if (ifToShare) {
    if (ifToLocalShare) {
      $('#screen-btn').addClass('selected disabled');
    } else {
      $('#screen-btn').addClass('disabled');
    }
    $('#galaxy-btn,#monitor-btn').addClass('disabled');
  } else {
    if (subscribeType === SUBSCRIBETYPES.FORWARD) {
      $('#galaxy-btn,#monitor-btn').removeClass('disabled');
    }
    $('#video-panel').removeClass('screen');
  }
}

// decide next size according to previous sizes and window w and h
function getNextSize() {
  var lSum = $('#video-panel .small').length * smallRadius * smallRadius * 5;
  var sSum = $('#video-panel .large').length * largeRadius * largeRadius * 10;
  var largeP = 1 / ((lSum + sSum) / $('#video-panel').width() / $(
    '#video-panel').height() + 1) - 0.5;
  if (Math.random() < largeP) {
    return 'large';
  } else {
    return 'small';
  }
}

function addVideo(stream, isLocal) {
  // compute next html id
  //var id = $('#video-panel').children('.client').length;
  var id = stream.id;
  console.log("addVideo video panel client length:", id);
  //while ($('#client-' + id).length > 0) {
    //++id;
  //}
  var uid = stream.origin;
  if (isLocal) {
    console.log("localStream addVideo1");
  }

  // check if is screen sharing
  if (stream.source.video === 'screen-cast') {
    $('#video-panel').addClass('screen')
      .append('<div class="client" id="screen"></div>');
    $('#screen').append('<video id="remoteScreen" playsinline autoplay class="palyer" style="width:100%;height:100%"></video>');
    $('#remoteScreen').get(0).srcObject = stream.mediaStream;
    // stream.show('screen');
    $('#screen').addClass('clt-' + getColorId(uid))
      .children().children('div').remove();
    $('#video-panel .largest').removeClass("largest");
    $('#screen').addClass("largest");
    $('#screen').append(
      '<div class="ctrl" id="original"><a href="#" class="ctrl-btn original"></a><a href="#" class="ctrl-btn enlarge"></a><a href="#" class="ctrl-btn ' +
      'fullscreen"></a></div>').append('<div class="ctrl-name">' +
      'Screen Sharing from ' + getUserFromId(stream.origin)["userId"] + '</div>');
    $('#local-screen').remove();
    changeMode(MODES.LECTURE, !isLocalScreenSharing);
    streamObj["screen"] = stream;
    $('#screen-btn').addClass('disabled');

  } else {
    // append to global users
    var thisUser = getUserFromId(uid) || {};
    var htmlClass = isLocal ? 0 : (id - 1) % 5 + 1;
    thisUser.htmlId = id;
    thisUser.htmlClass = thisUser.htmlClass || htmlClass;
    thisUser.id = uid;

    // append new video to video panel
    var size  = getNextSize();
    $('#video-panel').append('<div class="' + size + ' clt-' + htmlClass +
      ' client pulse" ' + 'id="client-' + id + '"></div>') ;
    if (isLocal) {
      $('#client-' + id).append('<div class="self-arrow"></div>');
      $('#client-' + id).append('<video id="localVideo" playsinline muted autoplay style="position:relative"></video>')
      $('#localVideo').get(0).srcObject = stream.mediaStream;
    }else {
      $('#client-' + id).append('<video id="remoteVideo" playsinline autoplay style="position:relative"></video>')
      $('#remoteVideo').get(0).srcObject = stream.mediaStream;
    }

    var hasLeft = mode === MODES.GALAXY,
      element = $("#client-" + id),
      width = element.width(),
      height = element.height();
    element.find("video").css({
      width: hasLeft ? "calc(100% + " + (4 / 3 * height - width) + "px)" : "100%",
      height: "100%",
      top: "0px",
      left: hasLeft ? -(4 / 3 * height / 2 - width / 2) + "px" : "0px"
    });
    var player = $('#client-' + id).children(':not(.self-arrow)');
    player.attr('id', 'player-' + id).addClass('player')
      .css('background-color', 'inherit');
    player.children('div').remove();
    player.attr('id', 'player-' + id).addClass('video');

    // add avatar for no video users
    if (stream.mediaStream === false) {
      player.parent().addClass('novideo');
      player.append('<img src="img/avatar.png" class="img-novideo" />');
    }

    // control buttons and user name panel
    var resize = size === 'large' ? 'shrink' : 'enlarge';
    if (stream.source.video === 'mixed'){
      var name = "Mix Stream";
      $('#video-panel .largest').removeClass("largest");
      $("#client-" + id).addClass("largest");
    }else {
      var name = (stream === localStream) ? localName : getUserFromId(stream.origin).userId || {};
    }
    var muteBtn = "";

    if (stream.source.audio === 'mixed' && stream.source.video === 'mixed') {
      name = "MIX Stream";
      stream.hide = null;
      $("#client-" + id).attr("isMix", "true");
      document.getElementById("player-" + id).ondblclick = null;
      $("#client-" + id).find("video").attr("stream", "mix");
      $("#client-" + id).find("video").dblclick(function(e){
        if($('#video-' + id).attr("stream") === "mix"){
          var width = $('#video-' + id).width();
          var height = width*scaleLevel;
          var offset = ($('#video-' + id).height()-height)/2;
          var left = (e.offsetX/width).toFixed(3);
          var top = ((e.offsetY-offset)/height).toFixed(3);
          var streamId = getStreamId(left, top);
          if (streamId && streamObj[streamId]) {
            room.subscribe(streamObj[streamId], function() {
              console.info('subscribed:', streamId);
              $('#video-' + id).attr("src", streamObj[streamId].createObjectURL());
              $('#video-' + id).attr("stream", streamId);
              // stopMonitor();
              // monitor(streamObj[streamId]);
            }, function(err) {
              console.error(streamId, 'subscribe failed:', err);
            });
            stream.signalOnPauseAudio();
            stream.signalOnPauseVideo();
          }
        } else {
          var forward = streamObj[$('#video-' + id).attr("stream")];
          if (forward) {
            stream.signalOnPlayVideo();
            stream.signalOnPlayAudio();
            $('#video-' + id).attr("src", stream.createObjectURL());
            $('#video-' + id).attr("stream", "mix");
            // stopMonitor();
            // monitor(stream);
            room.unsubscribe(forward, function(et) {
              console.info(forward.id(), 'unsubscribe stream');
            }, function(err) {
              console.error(stream.id(), 'unsubscribe failed:', err);
            });
          }
        }
      });
      muteBtn = '<a href="#" class="ctrl-btn unmute"></a>';
      player.parent().append('<div id="pause-' + id +
        '" class="pause" style="display: none; width: 100%; height: auto; position: absolute; text-align: center; font: bold 30px Arial;">Paused</canvas>'
      );
    }
    streamIndices['client-' + id] = stream.id;

    $('#client-' + id).append('<div class="ctrl">' +
        '<a href="#" class="ctrl-btn ' + resize + '"></a>' +
        '<a href="#" class="ctrl-btn fullscreen"></a>' + muteBtn + '</div>')
      .append('<div class="ctrl-name">' + name + '</div>').append(
        "<div class='noCamera'></div>");
    stream.addEventListener('ended', function(event) {
      console.log("====stream ended:", stream.id);
      $('#client-' + stream.id).remove();
      if (stream.source.video === 'screen-cast') {
        $('#screen-btn').removeClass('disabled');
      }
    });
    relocate($('#client-' + id));
    changeMode(mode);
  }

  function mouseout(e) {
    isMouseDown = false;
    mouseX = null;
    mouseY = null;
    $(this).css('transition', '0.5s');
  }
  // no animation when dragging
  $('.client').mousedown(function(e) {
      isMouseDown = true;
      mouseX = e.clientX;
      mouseY = e.clientY;
      $(this).css('transition', '0s');
    }).mouseup(mouseout).mouseout(mouseout)
    .mousemove(function(e) {
      e.preventDefault();
      if (!isMouseDown || mouseX === null || mouseY === null || mode !==
        MODES.GALAXY) {
        return;
      }
      // update position to prevent from moving outside of video-panel
      var left = parseInt($(this).css('left')) + e.clientX - mouseX;
      var border = parseInt($(this).css('border-width')) * 2;
      var maxLeft = $('#video-panel').width() - $(this).width() - border;
      if (left < 0) {
        left = 0;
      } else if (left > maxLeft) {
        left = maxLeft;
      }
      $(this).css('left', left);

      var top = parseInt($(this).css('top')) + e.clientY - mouseY;
      var maxTop = $('#video-panel').height() - $(this).height() - border;
      if (top < 0) {
        top = 0;
      } else if (top > maxTop) {
        top = maxTop;
      }
      $(this).css('top', top);

      // update data for later calculation position
      $(this).data({
        left: left,
        top: top
      });
      mouseX = e.clientX;
      mouseY = e.clientY;
    });

  // stop pulse when animation completes
  setTimeout(function() {
    $('#client-' + id).removeClass('pulse');
  }, 800);
}

function getStreamId(left, top) {
  for(var i in currentRegions) {
    if(left > currentRegions[i].left && left < (currentRegions[i].left + currentRegions[i].relativeSize) && top > currentRegions[i].top && top < (currentRegions[i].top + currentRegions[i].relativeSize)) {
      return currentRegions[i].streamID;
    }
  }
  return null;
}

function toggleIm(isToShow) {
  if (isToShow || $('#text-panel').is(":visible")) {
    $('#video-panel').css('left', 0);
  } else {
    $('#video-panel').css('left', $('#text-panel').width());
  }
  $('#text-panel').toggle();
  if ($('#im-btn').hasClass('selected')) {
    $('#im-btn').removeClass('selected');
  } else {
    $('#im-btn').addClass('selected');
  }
  changeMode(mode);
}

function relocate(element) {
  var max_loop = 1000;
  var margin = 20;
  for (var loop = 0; loop < max_loop; ++loop) {
    var r = element.hasClass('large') ? largeRadius : smallRadius;
    var w = $('#video-panel').width() - 2 * r - 2 * margin;
    var y = $('#video-panel').height() - 2 * r - 2 * margin;
    var x = Math.ceil(Math.random() * w + r + margin);
    var y = Math.ceil(Math.random() * y + r + margin);
    var others = $('.client');
    var len = others.length;
    var isFeasible = x + r < $('#video-panel').width() && y + r < $(
      '#video-panel').height();
    for (var i = 0; i < len && isFeasible; ++i) {
      var o_r = $(others[i]).hasClass('large') ? largeRadius : smallRadius;
      var o_x = parseInt($(others[i]).data('left')) + o_r;
      var o_y = parseInt($(others[i]).data('top')) + o_r;
      if ((o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) <
        (o_r + r + margin) * (o_r + r + margin)) {
        // conflict
        isFeasible = false;
        break;
      }
    }
    if (isFeasible) {
      var pos = {
        'left': x - r,
        'top': y - r
      };
      element.css(pos).data('left', x - r).data('top', y - r);
      return true;
    }
  }
  // no solution
  var pos = {
    'left': x - r,
    'top': y - r
  };
  element.css(pos).data('left', x - r).data('top', y - r);
  return false;
}

function sendIm(msg, sender) {
  var time = new Date();
  var hour = time.getHours();
  hour = hour > 9 ? hour.toString() : '0' + hour.toString();
  var mini = time.getMinutes();
  mini = mini > 9 ? mini.toString() : '0' + mini.toString();
  var sec = time.getSeconds();
  sec = sec > 9 ? sec.toString() : '0' + sec.toString();
  var timeStr = hour + ':' + mini + ':' + sec;
  if (msg === undefined) {
    // send local msg
    if ($('#text-send').val()) {
      msg = $('#text-send').val();
      var sendMsgInfo = JSON.stringify({
        type: "msg",
        data: msg
      })
      $('#text-send').val('').height('18px');
      $('#text-content').css('bottom', '30px');
      sender = localId;
      console.info('ready to send message');
      // send to server
      if (localName !== null) {
        room.send(sendMsgInfo).then(() => {
          console.info('begin to send message');
          console.info(localName + 'send message: ' + msg);
        }, err => {
          console.error(localName + 'sned failed: ' + err);
        });
      }
    } else {
      return;
    }
  }

  var color = getColor(sender);
  var user = getUserFromId(sender);
  var name = user ? user['userId'] : 'System';
  if (name !== 'System') {
    $('<p class="' + color + '">').html(timeStr + ' ' + name + '<br />')
      .append(document.createTextNode(msg)).appendTo('#text-content');
    // scroll to bottom of text content
    $('#text-content').scrollTop($('#text-content').prop('scrollHeight'));
  }
}

function getColorId(id) {
  var user = getUserFromId(id);
  if (user) {
    return user.htmlClass;
  } else {
    // screen stream comes earlier than video stream
    var htmlClass = users.length % 5 + 1;
    users.push({
      name: name,
      htmlClass: htmlClass
    });
    return htmlClass;
  }
}

function getColor(id) {
  var user = getUserFromId(id);
  if (user) {
    return 'clr-clt-' + user.htmlClass;
  } else {
    return 'clr-sys';
  }
}

function getUserFromName(name) {
  for (var i = 0; i < users.length; ++i) {
    if (users[i] && users[i].userId === name) {
      return users[i];
    }
  }
  return null;
}

function getUserFromId(id) {
  for (var i = 0; i < users.length; ++i) {
    if (users[i] && users[i].id === id) {
      return users[i];
    }
  }
  return null;
}

function deleteUser(id) {
  var index = 0;
  for (var i = 0; i < users.length; ++i) {
    if (users[i] && users[i].id === id) {
      index = i;
      break;
    }
  }
  users.splice(index, 1);
  $('li').remove(":contains(" + id + ")");  
}

function toggleMute(id, toMute) {
  if (streamObj[streamIndices["client-" + id]]) {
    if (toMute) {
      streamObj[streamIndices["client-" + id]].disableAudio();
    } else {
      streamObj[streamIndices["client-" + id]].enableAudio();
    }
  }
}

function getColumns() {
  var col = 1;
  var cnt = $('#video-panel video').length;
  if (mode === MODES.LECTURE && !isScreenSharing) {
    --cnt;
  }
  if (cnt === 0) {
    return 0;
  }
  while (true) {
    var width = mode === MODES.MONITOR ?
      Math.floor($('#video-panel').width() / col) :
      Math.floor($('#text-panel').width() / col);
    var height = Math.floor(width * 3 / 4);
    var row = Math.floor($('#video-panel').height() / height);
    if (row * col >= cnt) {
      return col;
    }
    ++col;
  }
}

function changeMode(newMode, enlargeElement) {
  if (localStream) {
    console.log("localStream changeMode" + newMode);
  }
  switch (newMode) {
    case MODES.GALAXY:
      if ($('#galaxy-btn').hasClass('disabled')) {
        return;
      }
      mode = MODES.GALAXY;
      if (subscribeType === SUBSCRIBETYPES.FORWARD) {
        $('#galaxy-btn,#monitor-btn').removeClass('selected');
      } else {
        $('#galaxy-btn,#monitor-btn').addClass('disabled');
      }
      $('#galaxy-btn').addClass('selected');
      $('#video-panel').removeClass('monitor lecture')
        .addClass('galaxy');
      $.each($('.client'), function(key, value) {
        var d = smallRadius * 2;
        if ($(this).hasClass('large')) {
          d = largeRadius * 2;
          $(this).find('.enlarge')
            .removeClass('enlarge').addClass('shrink');
        } else {
          $(this).find('.shrink')
            .removeClass('shrink').addClass('enlarge');
        }
        var left = parseInt($(this).data('left'));
        if (left < 0) {
          left = 0;
        } else if (left > $('#video-panel').width() - d) {
          left = $('#video-panel') - d;
        }
        var top = parseInt($(this).data('top'));
        if (top < 0) {
          top = 0;
        } else if (top > $('#video-panel').height() - d) {
          top = $('#video-panel').height() - d;
        }
        $(this).css({
          left: left,
          top: top,
          width: d + 'px',
          height: d + 'px'
        }).data({
          left: left,
          top: top
        });
      });
      setTimeout(function() {
        $('.client').css("position", "absolute");
      }, 500);
      break;

    case MODES.MONITOR:
      if ($('#monitor-btn').hasClass('disabled')) {
        return;
      }
      mode = MODES.MONITOR;
      if (subscribeType === SUBSCRIBETYPES.FORWARD) {
        $('#galaxy-btn,#monitor-btn').removeClass('selected');
      } else {
        $('#galaxy-btn,#monitor-btn').addClass('disabled');
      }
      $('#monitor-btn').addClass('selected');
      $('#video-panel').removeClass('galaxy lecture')
        .addClass('monitor');
      $('.shrink').removeClass('shrink').addClass('enlarge');
      // updateMonitor();
      break;

    case MODES.LECTURE:
      if ($('#lecture-btn').hasClass('disabled')) {
        return;
      }
      mode = MODES.LECTURE;
      if (subscribeType === SUBSCRIBETYPES.FORWARD) {
        $('#galaxy-btn,#monitor-btn').removeClass('selected');
      } else {
        $('#galaxy-btn,#monitor-btn').addClass('disabled');
      }
      $('#lecture-btn').addClass('selected');
      $('#video-panel').removeClass('galaxy monitor')
        .addClass('lecture');
      $('.shrink').removeClass('shrink').addClass('enlarge');
      if (typeof enlargeElement !== 'boolean') {
        var largest = enlargeElement || ($('#screen').length > 0 ? $('#screen') :
          ($('.largest').length > 0 ? $('.largest').first() : ($('.large').length >
            0 ? $('.large').first() : $('.client').first())));
        $('.client').removeClass('largest');
        largest.addClass('largest')
          .find('.enlarge').removeClass('enlarge').addClass('shrink');
      }
      updateLecture(enlargeElement);
      break;

    default:
      console.error('Illegal mode name');
  }
  //TODO: limit https
  if (window.location.protocol !== "https:") {
    // $("#screen-btn").addClass("disabled");
  }

  // update canvas size in all video panels
  $('.player').trigger('resizeVideo');
  setTimeout(resizeStream, 500, newMode);
}

function resizeStream(newMode) {
  if (!localStream) return;
  var hasLeft = newMode === MODES.GALAXY;
  for (var temp in streamObj) {
    //console.info(temp);
    var stream = streamObj[temp].id === localStream.id ? localStream :
      streamObj[temp],
      element = $("#client-" + temp),
      width = element.width(),
      height = element.height();
    if (stream.source.audio === 'screen-cast' && stream.source.video === 'screen-cast') {
      element.find("video").css({
        width: hasLeft ? "calc(100% + " + (4 / 3 * height - width) + "px)" :
          "" + stream.width,
        height: "" + stream.height,
        top: "0px",
        left: hasLeft ? -(4 / 3 * height / 2 - width / 2) + "px" : "0px"
      });
    } else {
      element.find("video").css({
        width: hasLeft ? "calc(100% + " + (4 / 3 * height - width) + "px)" :
          "100%",
        height: "100%",
        top: "0px",
        left: hasLeft ? -(4 / 3 * height / 2 - width / 2) + "px" : "0px"
      });
      if (element.find('.pause').length > 0) {
        element.find('.pause').css({
          width: hasLeft ? "calc(100% + " + (4 / 3 * height - width) +
            "px)" : "100%",
          height: "auto",
          top: height / 2.2 + "px",
          left: hasLeft ? -(4 / 3 * height / 2 - width / 2) + "px" : "0px"
        });
      }
      if (element.attr('isMix') === 'true') {
        $('#wifi').css('bottom', (height + scaleLevel * width)/2 < height ? (height + scaleLevel * width)/2 + 58 + 'px' : height + 58 + 'px');
      }
    }
  }
}

function updateMonitor() {
  var col = getColumns();
  if (col > 0) {
    $('.client').css({
      width: Math.floor($('#video-panel').width() / col),
      height: Math.floor($('#video-panel').width() / col * 3 / 4),
      top: 'auto',
      left: 'auto',
      position: "relative",
      right: "auto"
    });
  }
}

function updateLecture(hasChange) {
  if (typeof hasChange !== 'boolean') hasChange = true;
  $('.largest').css({
    width: $('#video-panel').width() - $('#text-panel').width(),
    height: $('#video-panel').height(),
    position: "absolute"
  });

  var col = isScreenSharing ? 1 : getColumns();
  var tempTop = 0;
  var tempRight = 0;
  if (!hasChange) return;
  $('.client').not('.largest').each(function(i) {
    if (i === 0) {
      tempTop = 0;
    } else if (i % col === 0) {
      tempTop += Math.floor($('#text-panel').width() / col * 3 / 4);
      tempRight = 0;
    } else {
      tempRight += Math.floor($('#text-panel').width() / col);
    }

    // if (subscribeType === SUBSCRIBETYPES.FORWARD) {
    $(this).css("position", "absolute");
    // }

    $(this).css({
      width: Math.floor($('#text-panel').width() / col),
      height: Math.floor($('#text-panel').width() / col * 3 / 4),
      right: tempRight,
      top: tempTop,
      left: "auto"
    });
  });
}

function fullScreen(isToFullScreen, element) {
  if (isToFullScreen) {
    element.addClass('full-screen');
  } else {
    element.removeClass('full-screen');
  }
}

function exitFullScreen(ctrlElement) {
  if (ctrlElement.parent().hasClass('full-screen')) {
    fullScreen(false, ctrlElement.parent());
    //        ctrlElement.find(".shrink").removeClass('shrink').addClass('enlarge');;
    //        ctrlElement.find(".unmute").before('<a href="#" class="ctrl-btn fullscreen">');
    if ((ctrlElement.parent().hasClass('small') || mode !== MODES.GALAXY) &&
      ctrlElement.parent().attr("id") == "screen") {
      ctrlElement.children('.shrink')
        .removeClass('shrink').addClass('fullscreen');
      if (isSmall) {
        ctrlElement.find(".fullscreen").before(
          '<a href="#" class="ctrl-btn enlarge">');
      }
    } else {
      ctrlElement.find(".shrink").removeClass('shrink').addClass('enlarge');;
      ctrlElement.find(".unmute").before(
        '<a href="#" class="ctrl-btn fullscreen">');
    }
    return;
  }
  switch (mode) {
    case MODES.GALAXY:
      ctrlElement.children('.shrink')
        .addClass('enlarge').removeClass('shrink').parent()
        .parent().removeClass('large').addClass('small');
      break;

    case MODES.MONITOR:
    case MODES.LECTURE:
      changeMode(MODES.LECTURE);
      break;
  }
}

// no use
function playpause() {
  var el = event.srcElement;
  if (el.getAttribute("isPause") != undefined) {
    el.innerText = "Pause Video";
    for (var tmp in room.remoteStreams) {
      var stream = room.remoteStreams[tmp];
      if (stream.id() !== localStream.id()) {
        stream.playVideo();
      } else {
        localStream.enableVideo();
      }
    }
    $(".noCamera").hide();
    el.removeAttribute("isPause");
  } else {
    el.innerText = "Play Video";
    for (var tmp in room.remoteStreams) {
      var stream = room.remoteStreams[tmp];
      if (stream.id() !== localStream.id()) {
        stream.pauseVideo();
      } else {
        localStream.disableVideo();
      }
    }
    $(".noCamera").show();
    el.setAttribute("isPause", "");
  }
}

function toggleVideo() {
  if (!localPublication || isAudioOnly) {
    return;
  }

  if (!isPauseVideo) {
    //TODO: pause all video?
     //remoteMixedSub.mute(Owt.Base.TrackKind.VIDEO);
    for (var temp in subList) {
      if (subList[temp] === screenSub) {
        continue;
      }
      subList[temp].mute(Owt.Base.TrackKind.VIDEO)
    }
    $('[ismix=true]').children('.video').css('display', 'none');
    $('[ismix=true]').children('.pause').css('display', 'block');
    localStream.mediaStream.getVideoTracks()[0].enabled = false;
    localPublication.mute(Owt.Base.TrackKind.VIDEO).then(
      () => {
        console.info('mute video');
        $('#pauseVideo').css({
          backgroundImage: 'url("img/turn-video.png")',
          backgroundColor: '#ccc'
        });
        isPauseVideo = !isPauseVideo;
        $('#promt').css('opacity', '0');
      }, err => {
        console.error('mute video failed');
      }
    );
  } else {
    $('[ismix=true]').children('.video').css('display', 'block');
    $('[ismix=true]').children('.pause').css('display', 'none');
     //remoteMixedSub.unmute(Owt.Base.TrackKind.VIDEO);
    for (var temp in subList) {
      if (subList[temp] === screenSub) {
        continue;
      }
      subList[temp].unmute(Owt.Base.TrackKind.VIDEO)
    }
    localStream.mediaStream.getVideoTracks()[0].enabled = true;
    localPublication.unmute(Owt.Base.TrackKind.VIDEO).then(
      () => {
        console.info('unmute video');
        $('#pauseVideo').css({
          backgroundImage: 'url("img/video.png")',
          backgroundColor: '#7bff7a'
        });
        isPauseVideo = !isPauseVideo;
      }, err => {
        console.error('unmute video failed');
      }
    );
  }
}

function toggleAudio() {
  if (!localPublication) {
    return;
  }

  if (!isPauseAudio) {
    localPublication.mute(Owt.Base.TrackKind.AUDIO).then(
      () => {
        console.info('mute successfully');
        $('#pauseAudio').css({
          backgroundImage: 'url("img/mute-voice.png")',
          backgroundColor: '#ccc'
        });
        isPauseAudio = !isPauseAudio;
      },err => {
        console.error('mute failed');
        $('#pauseAudio').css({
          backgroundImage: 'url("img/audio.png")',
          backgroundColor: '#7bff7a'
        });
      }
    );
  } else {
    localPublication.unmute(Owt.Base.TrackKind.AUDIO).then(
      () => {
        console.info('unmute successfully');
        $('#pauseAudio').css({
          backgroundImage: 'url("img/audio.png")',
          backgroundColor: '#7bff7a'
        });
        isPauseAudio = !isPauseAudio;
      },err => {
        console.error('unmute failed');
        $('#pauseAudio').text("Unmute me");
      }
    );
  }
}

$(document).ready(function() {
  $('.buttonset>.button').click(function() {
    $(this).siblings('.button').removeClass('selected');
    $(this).addClass('selected');
  })

  // name
  if (window.location.search.slice(0, 6) === '?room=') {
    roomId = window.location.search.slice(6);
  }

  if (window.location.protocol === "https:") {
    $('#screen-btn').removeClass('disabled');
  } else {
    // $('#screen-btn').addClass('disabled');
  }

  $(document).on('click', '#pauseVideo', function() {
    toggleVideo();
  });

  $(document).on('click', '#pauseAudio', function() {
    toggleAudio();
  });

  $(document).on('click', '.original', function() {
    if (isOriginal) {
      $(this).parent().siblings().children('video').css('width', '100%');
      $(this).parent().siblings().children('video').css('height',
        '100%');
      $(this).parent().siblings().css('overflow', 'auto');
      //$(this).parent().siblings().children('video').parent().css('z-index','100');
      isOriginal = !isOriginal;
    } else {
      $(this).parent().siblings().children('video').css('width', '');
      $(this).parent().siblings().children('video').css('height', '');
      isOriginal = !isOriginal;
    }
  });

  $(document).on('click', '.shrink', function() {
    exitFullScreen($(this).parent());
    $(this).parent().parent().children('.player').trigger('resizeVideo');
    setTimeout(resizeStream, 500, mode);
  });

  $(document).on('click', '.enlarge', function() {
    switch (mode) {
      case MODES.GALAXY:
        $(this).addClass('shrink').removeClass('enlarge').parent()
          .parent().removeClass('small').addClass('large');
        break;

      case MODES.MONITOR:
      case MODES.LECTURE:
        changeMode(MODES.LECTURE, $(this).parent().parent());
        break;
    }
    $(this).parent().parent().children('.player').trigger('resizeVideo');
    setTimeout(resizeStream, 500, mode);
  });

  $(document).on('click', '.mute', function() {
    // unmute
    var id = parseInt($(this).parent().parent().attr('id').slice(7));
    toggleMute(id, false);
    $(this).addClass('unmute').removeClass('mute');
  });

  $(document).on('click', '.unmute', function() {
    // mute
    var id = parseInt($(this).parent().parent().attr('id').slice(7));
    toggleMute(id, true);
    $(this).addClass('mute').removeClass('unmute');
  });

  //TODO:mute others
  $(document).on('dblclick', '.muteShow', function() {
    // mute others
    var muteId = $(this).siblings('.userID').text();
    var msg = {
      type: "force",
    };
    forwardStreamMap.forEach((stream, key) => {
      if (stream.info.owner === muteId && stream.media.audio) {
        if (stream.media.audio.status === 'active') {
          pauseStream(roomId, stream.id, 'audio', () => chgMutePic(muteId, true));
        } else {
          playStream(roomId, stream.id, 'audio', () => chgMutePic(muteId, false));
        }
      }
    });
  });

  $(document).on('click', '.fullscreen', function() {
    fullScreen(true, $(this).parent().parent());
    var enlarge = $(this).siblings('.enlarge');
    if (enlarge.length > 0) {
      enlarge.removeClass('enlarge').addClass('shrink');
      isSmall = true;
    } else if ($(this).siblings('.shrink').length === 0) {
      var unmute = $(this).parent().find(".unmute");
      if (unmute.length > 0) {
        unmute.before('<a href="#" class="ctrl-btn shrink"></a>');
      } else {
        $(this).parent().append(
          '<a href="#" class="ctrl-btn shrink"></a>');
      }
      isSmall = false;
    }
    $(this).remove();
  });

  $(document).keyup(function(event) {
    if (event.keyCode === 27 && $('.full-screen').length > 0) {
      console.log('full');
      // exit full screen when escape key pressed
      exitFullScreen($('.full-screen .ctrl'));
    }
  });

  $('#text-send').keypress(function(event) {
    if ($(this)[0].scrollHeight > $(this)[0].clientHeight) {
      $(this).height($(this)[0].scrollHeight);
      $('#text-content').css('bottom', $(this)[0].scrollHeight + 'px');
    }
    if (event.keyCode === 13) {
      event.preventDefault();
      // send msg when press enter
      sendIm();
    }
  });

  $('#login-input').keypress(function(event) {
    if (event.keyCode === 13) {
      event.preventDefault();
      login();
    }
  });

  $('.dialog--close').click(function() {
    $(this).parent().parent().hide();
  });

  $('#login-panel').fadeIn();

  $(window).resize(function() {
    console.log('resized');
    changeMode(mode);
  });

  checkMobile();
});

$(window).unload(function() {
  userExit();
});

function checkMobile() {
  if ((/iphone|ipod|android|ie|blackberry|fennec/).test(navigator.userAgent.toLowerCase())) {
    isMobile = true;
    $("#im-btn").hide();
    toggleIm(true);
    changeMode(MODES.MONITOR);
    $('#galaxy-btn, .button-split, #screen-btn').hide();
  }
}

