# Turbine polynomial

This project contains data cleaning and machine learning tools for (supervised)
learning a polynomial function based on turbine characteristics.

The accessible input data consists of
* `AIR_IN_TEMP`
* `AIR_IN_PRES`
* `DISCHARGE_TEMP`
* `DISCHARGE_PRES`
* `SPEED`
* `TURBINE_TYPE`

and the learned output is the `SIMULATED_EFF` feature.

To run the program, use

```bash
bin/turbine --method=fcv data/*
```

and use `bin/turbine --help` for more information about the arguments.

This project also offers a library for preprocessing and learning the data, and
can be used in Python by importing `turbinelearn`.
