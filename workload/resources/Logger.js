class Logger {
  constructor($dom) {
    this.$dom = $dom;
    this.indent = 0;
  }

  log = (message) => {
    console.log(message);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + message}`;
  };

  error = (err) => {
    console.error(err);
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + err.message}`;
  };

  group = (name) => {
    console.group(name);
    this.log('');
    this.$dom.innerHTML += `\n${'\t'.repeat(this.indent) + name}`;
    this.indent++;
  };

  groupEnd = () => {
    console.groupEnd();
    this.indent--;
  };
}