
import pandas as pd


table = {
    "name" : ["Ali", "Ahmed", "Hassan", "Hussain", "Ali"],
    "age" : [25, 30, 35, 40, 45],
    "city" : ["Karachi", "Lahore", "Islamabad", "Karachi", "Lahore"]
}

df = pd.DataFrame(table)

print(df[["name"]].to_dict())
