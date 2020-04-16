const getTime = () => {
  const time = new Date()
  let hour = time.getHours()
  hour = hour > 9 ? hour.toString() : '0' + hour.toString()
  let mini = time.getMinutes()
  mini = mini > 9 ? mini.toString() : '0' + mini.toString()
  let sec = time.getSeconds()
  sec = sec > 9 ? sec.toString() : '0' + sec.toString()
  return hour + ':' + mini + ':' + sec
}

export default getTime
