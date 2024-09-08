from consolemenu import *
from consolemenu.items import *
from idna.idnadata import scripts

import scripts.get_failure_serial_numbers
import scripts.get_small_dataset
import scripts.preparing_dataset
import scripts.train_ML_regression
import scripts.predict_ML_regression

pu = PromptUtils(Screen())

def run_get_failure_serial_numbers():
    scripts.get_failure_serial_numbers.run()
    pu.enter_to_continue()


def run_get_small_dataset():
    scripts.get_small_dataset.run()
    pu.enter_to_continue()


def run_preparing_dataset():
    scripts.preparing_dataset.run()
    pu.enter_to_continue()

def run_train_ml_regression():
    scripts.train_ML_regression.run()
    pu.enter_to_continue()

def run_predict_ml_regression():
    scripts.predict_ML_regression.run()
    pu.enter_to_continue()

if __name__ == '__main__':
    # Create the root menu
    menu = ConsoleMenu("Sila HDD")

    # Create a menu item that calls a function
    function_item1 = FunctionItem("Get failure serial numbers", run_get_failure_serial_numbers)
    function_item2 = FunctionItem("Get Small Dataset", run_get_small_dataset)
    function_item3 = FunctionItem("Preparing dataset", run_preparing_dataset)
    function_item4 = FunctionItem("Train ML regression", run_train_ml_regression)
    function_item5 = FunctionItem("Predict ML regression", run_predict_ml_regression)

    # Add all the items to the root menu
    menu.append_item(function_item1)
    menu.append_item(function_item2)
    menu.append_item(function_item3)
    menu.append_item(function_item4)
    menu.append_item(function_item5)

    # Show the menu
    menu.start()
    menu.join()