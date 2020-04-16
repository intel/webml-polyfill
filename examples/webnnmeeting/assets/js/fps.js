const Stats = function() {
  let beginTime = (performance || Date).now()
  let prevTime = beginTime
  let frames = 0
  const fpsPanel = new Stats.Panel()
  return {
    begin() {
      beginTime = (performance || Date).now()
    },
    end() {
      frames++
      const time = (performance || Date).now()
      if (time > prevTime + 1000) {
        fpsPanel.update((frames * 1000) / (time - prevTime), 100)
        prevTime = time
        frames = 0
      }
      return time
    },
    update() {
      beginTime = this.end()
    }
  }
}

// eslint-disable-next-line import/no-mutable-exports
let fps = 0

Stats.Panel = function() {
  let min = Infinity
  let max = 0
  const round = Math.round
  return {
    update(value, maxValue) {
      min = Math.min(min, value)
      max = Math.max(max, value)
      fps = round(value)
    }
  }
}

export { Stats, fps }
