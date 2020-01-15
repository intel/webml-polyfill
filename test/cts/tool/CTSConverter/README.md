# CTSConverter
Transfer [nn (tag: android-cts-10.0_r2)](https://android.googlesource.com/platform/frameworks/ml/+/refs/tags/android-cts-10.0_r2) test case to [webml](https://github.com/intel/webml-polyfill) test case

## Prerequisites
* Need `python3`

## Start

Update `../../test/cts-all.js` and `../../test_supplement/cts_supplement-all.js` files.
`../../test/cts-all.js` includes test cases in `../../test` directory and its sub directory, `../../test_supplement/cts_supplement-all.js` includes test cases in `../../test_supplement` directory.

```shell
$ npm start
```

## Transfer nn test case into `./output/cts`

```shell
$ npm run transfer
```

## Get more information for command line

```shell
$ npm run info
```

```shell
-h, --help            show this help message and exit
-a, --all             [option] '-a [all-file in relative directory]', create all test file
-t, --transfer        [option] '-t [transfer relative directory]', transfer nn test file
-c, --cts             [option] '-c [cts relative directory]', include cts test file
-s, --supplement      [option] '-s [supplement relative directory]', include supplement test file
-p, --plus            [option] '-p [plus relative directory]', include plus test file
```

## Example

1. Transfer nn test cases from `./src/nn/specs` and output into `./output/cts`.

```shell
$ python3 ./src/main.py -t ./src/nn/specs
```

2. Create all test cases file as `../../test_supplement/cts_supplement-all.js` from `../../test_supplement`.

```shell
$ python3 ./src/main.py -s ../../test_supplement -a ../../test_supplement/cts_supplement-all.js
```

3. Create all test cases file as `../../test/cts-all.js` from `../../test`.

```shell
$ python3 ./src/main.py -c ../../test -a ../../test/cts-all.js
```
