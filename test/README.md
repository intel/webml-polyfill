# Contribute CTS test case
The CTS tests in `./cts/test` folder were converted from original Android CTS test, for new created CTS test file, please add them into `./cts/test_supplement` folder

## Steps
Here are steps to update `./cts/test_supplement/cts_supplement-all.js`:
* Copy your test file into `./cts/test_supplement`.
* Change directory into `CTSConverter`:

    ```shell
    $ cd ./cts/tool/CTSConverter
    ```

* Update `./cts/test_supplement/cts_supplement-all.js` file:

    ```shell
    $ npm run combine
    ```

    or

    ```shell
    $ python3 ./src/main.py -s ../../test_supplement -o ../../test_supplement/cts_supplement-all.js
    ```
