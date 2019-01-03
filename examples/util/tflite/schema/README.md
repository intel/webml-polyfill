The schema_generated.js is compiled by schema compiler from https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/schema
which is licensed under the Apache License, Version 2.0.



# How to generate

## Prerequisites
Make sure you have installed `CMAKE` beforehand with the system package manager. 


## Steps
### Build flatc from source
```
$ git clone https://github.com/google/flatbuffers.git
$ cd flatbuffers
$ cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
$ make
$ sudo make install
```

### Download source code
```
$ git clone https://github.com/tensorflow/tensorflow.git
$ cd tensorflow/tensorflow/lite/schema
```

### Generate schema_generated.js
```
$ flatc --js schema.fbs
```