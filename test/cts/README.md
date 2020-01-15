# Contribute CTS test case
The tests in `./test/cts/test` folder were converted from original Android CTS test, for new created CTS test file, please add them into `./test/cts/test_supplement` folder

## Steps
Here are steps to update `./test/cts/test_supplement/cts_supplement-all.js`:
* Copy your test file into `./test/cts/test_supplement`.
* Change directory into `CTSConverter`:

    ```shell
    $ cd ./test/cts/tool/CTSConverter
    ```

* Update `./test/cts/test_supplement/cts_supplement-all.js` file:

    ```shell
    $ npm start
    ```

    or

    ```shell
    $ python3 ./src/main.py -s ../../test_supplement -a ../../test_supplement/cts_supplement-all.js
    ```
