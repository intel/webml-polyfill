// Copyright (C) <2018> Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

'use strict';

import * as base from './base/export.js';
import * as conference from './conference/export.js';

/**
 * Base objects for both P2P and conference.
 * @namespace Owt.Base
 */

const Base = base

/**
 * WebRTC connections with conference server.
 * @namespace Owt.Conference
 */
const Conference = conference

const Owt = {
  Base, Conference
}

export default Owt