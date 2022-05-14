import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model as lm

def average_list(bridge):
    return sum(bridge) // len(bridge) # via integer division ( // )

def average_three_ints(int1, int2, int3):
    return (int1 + int2 + int3) // 3

def perc_err(model, actual):
    return (model - actual) / actual
''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
# BEGIN GIVEN CODE
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')

dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))

# END GIVEN CODE
dataset_1['Total'] = pandas.to_numeric(dataset_1['Total'].replace(',', '', regex=True))
dataset_1['High Temp'] = pandas.to_numeric(dataset_1['High Temp'].replace(',', '', regex=True))
dataset_1['Low Temp'] = pandas.to_numeric(dataset_1['Low Temp'].replace(',', '', regex=True))
dataset_1['Precipitation'] = pandas.to_numeric(dataset_1['Precipitation'].replace(',', '', regex=True))

brooklyn = dataset_1['Brooklyn Bridge'].tolist()
manhattan = dataset_1['Manhattan Bridge'].tolist()
queensboro = dataset_1['Queensboro Bridge'].tolist()
williamsburg = dataset_1['Williamsburg Bridge'].tolist()

high_t = dataset_1['High Temp'].tolist()
low_t = dataset_1['Low Temp'].tolist()
prec = dataset_1['Precipitation'].tolist()
avg_t = [] # will hold average temps of each day in spreadsheet
# populate vector
for i in range(len(high_t)):
    avg_t.append((high_t[i] + low_t[i]) / 2)

### REQUIRED HISTOGRAM FOR DESCRIPTIVE STATISTICS - QUESTION 3
# AVERAGE TEMP HISTOGRAM
# use avg_t made above which will also be used in Question 2
temp_array =np.array(avg_t)
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(temp_array, bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlabel('Average Temperature (degrees F)')
plt.ylabel('Total Count')
plt.xlim([20, 100])
plt.title('Histogram of Count of Days with Average Temperature Range in NYC')

plt.show()

### QUESTION 1
print("QUESTION 1: \n")
# find bridge averages
b_avg = average_list(brooklyn)
m_avg = average_list(manhattan)
q_avg = average_list(queensboro)
w_avg = average_list(williamsburg)

# for each bridge, print its average then averages of other 3 bridges
# used in table 1
print('Brooklyn average:', b_avg, 'M/Q/W average:', average_three_ints(m_avg, q_avg, w_avg))
print('Manhattan average:', m_avg, 'B/Q/W average:', average_three_ints(b_avg, q_avg, w_avg))
print('Queensboro average:', q_avg, 'B/M/W average:', average_three_ints(b_avg, m_avg, w_avg))
print('Willaimsburg average:', w_avg, 'B/M/Q average:', average_three_ints(b_avg, m_avg, q_avg))
print("")
total = dataset_1['Total'].tolist()
date = []
for i in range(214):
    date.append(i+1)
# print(dataset_1.to_string()) #This line will print out your data
print("MODEL ESTIMATED VALUES FOR 4 RANDOM DATES")
print("April 1: ", 4*average_three_ints(brooklyn[0], manhattan[0], williamsburg[0]))
print("May 31: ", 4*average_three_ints(brooklyn[60], manhattan[60], williamsburg[60]))
print("August 2: ", 4*average_three_ints(brooklyn[123], manhattan[123], williamsburg[123]))
print("September 4: ", 4*average_three_ints(brooklyn[156], manhattan[156], williamsburg[156]))

# plot actual totals vs time
# plot predicted totals vs time
predicted = []
for i in range(214):
    predicted.append(4*average_three_ints(brooklyn[i], manhattan[i], williamsburg[i]))

# plot for q1
plt.plot(date, total, 'b*', label="Actual Total") # given values
plt.plot(date, predicted, 'y*', label="Predicted Total") # model values
# formatting (axis, titles)
plt.xlabel('Days Since April 1')
plt.ylabel('Total Bikes')
plt.title('Bikes Traffic Since April 1 in NYC')
plt.legend(loc='upper left')
plt.show()

plt.plot()

print("\nEND OF QUESTION 1")

### QUESTION 2
print("\nQUESTION 2:\n")

# plots for q2
# bikes vs avg temp
plt.plot(avg_t, total, 'g*', label="Actual") # true values
# calculate linear model
linear_model = np.polyfit(avg_t, total, 1)
lin_fn = np.poly1d(linear_model)
print("The equation for bikes vs average temperature is: y =", lin_fn, '\n')
plt.plot(avg_t, lin_fn(avg_t), label="Linear Model") # plot model
# formatting
plt.xlabel('Average Temperature (degrees F)')
plt.ylabel('Total Bikes')
plt.title('Bike Traffic vs Average Temperature in NYC')
plt.legend(loc='upper left')
plt.show()

# bikes vs precipitation
plt.plot(prec, total, 'g*', label="Actual") # real values
# calc linear model
linear_model = np.polyfit(prec, total, 1)
lin_fn_2 = np.poly1d(linear_model)
print("The equation for bikes vs precipitation is: y =", lin_fn_2, '\n')
plt.plot(prec, lin_fn_2(prec), label="Linear Model") # plot model
# formatting
plt.xlabel('Precipitation (mm)')
plt.ylabel('Total Bikes')
plt.title('Bike Traffic vs Precipitation in NYC')
plt.legend(loc='upper left')
plt.show()

# multivariable regression
indep_vars = [avg_t, prec] # holds average temperature and preciptation values
feature_mat = np.array(indep_vars)
target_vec = np.array(total)
feature_mat = feature_mat.transpose() # transpose to make temp and prec columns instead of rows
regr = lm.LinearRegression()
regr.fit(feature_mat, target_vec)
print("The coefficients for the average temperature and precipitation variables, respectively:\n", regr.coef_)
print("The y-intercept is:", regr.intercept_)

print("\nMODEL ESTIMATED VALUES FOR 4 RANDOM DATES")
print("April 6: ", regr.predict([[avg_t[5], prec[5]]]))
print("May 17: ", regr.predict([[avg_t[46], prec[46]]]))
print("June 7: ", regr.predict([[avg_t[67], prec[67]]]))
print("August 30: ", regr.predict([[avg_t[151], prec[151]]]))
print("\nMODEL ESTIMATED PERCENT ERROR VALUES FOR 4 RANDOM DATES")
print("April 6 % Error: ", abs(100*perc_err(regr.predict([[avg_t[5], prec[5]]]), total[5])))
print("May 17 % Error: ", abs(100*perc_err(regr.predict([[avg_t[46], prec[46]]]), total[46])))
print("June 7 % Error: ", abs(100*perc_err(regr.predict([[avg_t[67], prec[67]]]), total[67])))
print("August 30 % Error: ", abs(100*perc_err(regr.predict([[avg_t[151], prec[151]]]), total[151])))

tot_error = 0

for i in range(len(total)):
    tot_error = tot_error + 100 * perc_err(regr.predict([[avg_t[i], prec[i]]]), total[i])
avg_error = tot_error / len(total)
print("\nAverage error for all data:", avg_error)

print("\nEND OF QUESTION 2")

### QUESTION 3
print("\nQUESTION 3:\n")

# index is: 0mm, <0.25mm, <0.5mm, <0.75mm, >=0.75mm
#less5 = [0, 0, 0, 0, 0]
less10 = [0, 0, 0, 0, 0]
less15 = [0, 0, 0, 0, 0]
less20 = [0, 0, 0, 0, 0]
less25 = [0, 0, 0, 0, 0]
more25 = [0, 0, 0, 0, 0]
for i in range(len(total)):
    if total[i] < 10000:
        if prec[i] == 0:
            less10[0] += 1
        elif prec[i] < 0.25:
            less10[1] += 1
        elif prec[i] < 0.05:
            less10[2] += 1
        elif prec[i] < 0.75:
            less10[3] += 1
        else:
            less10[4] += 1
    elif total[i] < 15000:
        if prec[i] == 0:
            less15[0] += 1
        elif prec[i] < 0.25:
            less15[1] += 1
        elif prec[i] < 0.05:
            less15[2] += 1
        elif prec[i] < 0.75:
            less15[3] += 1
        else:
            less15[4] += 1
    elif total[i] < 20000:
        if prec[i] == 0:
            less20[0] += 1
        elif prec[i] < 0.25:
            less20[1] += 1
        elif prec[i] < 0.05:
            less20[2] += 1
        elif prec[i] < 0.75:
            less20[3] += 1
        else:
            less20[4] += 1
    elif total[i] < 25000:
        if prec[i] == 0:
            less25[0] += 1
        elif prec[i] < 0.25:
            less25[1] += 1
        elif prec[i] < 0.05:
            less25[2] += 1
        elif prec[i] < 0.75:
            less25[3] += 1
        else:
            less25[4] += 1
    else:
        if prec[i] == 0:
            more25[0] += 1
        elif prec[i] < 0.25:
            more25[1] += 1
        elif prec[i] < 0.05:
            more25[2] += 1
        elif prec[i] < 0.75:
            more25[3] += 1
        else:
            more25[4] += 1

print("Less than 10,000 bikers:", less10)
print("10,000 to 14,999 bikers:", less15)
print("15,000 to 19,999 bikers:", less20)
print("20,000 to 24,999 bikers:", less25)
print("25,000+ bikers:", more25)


print("\nEND OF QUESTION 3")


