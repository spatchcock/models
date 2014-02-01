# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 19:57:11 2013

@author: spatchcock
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
# The animation pattern is based on the animation tutorial found here: 
#  http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
#

class Economy:
        
    def __init__(self):
              
        # macro variables
        self.initial_monthly_spend = 1
        self.velocity_of_money     = 2
        
        # Flows
        self.government_spending_rate = 0.105
        self.government_taxation_rate = 0.05
        self.private_saving_rate      = 0.05
        
        # time series records
        self.government_spending_record  = [0.0]
        self.government_taxation_record  = [0.0]
        self.private_sector_income       = [0.0]
        self.private_sector_saving       = [0.0]
        self.gdp_record                  = [0.0]
        self.money_in_circulation        = [0.0]
                
        self.dt = 1.0/12.0  # monthly


    def spend(self):
        
        if len(self.private_sector_income) > 1:
            monthly_spend = self.private_sector_income[-1]*self.government_spending_rate
        else:
            monthly_spend = self.initial_monthly_spend
            
        self.government_spending_record.append(monthly_spend)        
        

    def tax(self):
        if len(self.private_sector_income) > 1:
            monthly_tax = self.private_sector_income[-1]*self.government_taxation_rate
        else:
            monthly_tax = 0.0            
        
        self.government_taxation_record.append(monthly_tax)
       

    def government_balance_record(self,t=None):
        if (t == None):
            t = -1
        
        acc = [None] * len(self.government_spending_record)
        
        for i in range(0, len(self.government_spending_record)):
            acc[i] = -self.government_spending_record[i] + self.government_taxation_record[i]
        
        return acc

    def private_sector_balance_record(self,t=None):
        return [monthly_balance*-1 for monthly_balance in self.government_balance_record(t)]
          

    def cumulative_private_sector_account(self,t=None):
        balance_record = self.private_sector_balance_record()
        cum_balance = [None] * len(self.government_spending_record)
        cum_balance[0] = balance_record[0]
        for i in range(1, len(balance_record)):
            cum_balance[i] = cum_balance[i-1] + balance_record[i]
        
        return cum_balance


    def cumulative_government_account(self,t=None):
        balance_record = self.government_balance_record(t)
        cum_balance = [None] * len(self.government_spending_record)
        cum_balance[0] = balance_record[0]
        for i in range(1, len(balance_record)):
            cum_balance[i] = cum_balance[i-1] + balance_record[i]
        return cum_balance               
        
    
    def recent_gdp(self):
        last_year = self.private_sector_balance_record[-12:]
        return sum(last_year) * self.velocity_of_money

    def grow(self):
        this_monthly_account = self.cumulative_private_sector_account()[-1]
        this_monthly_gdp = (this_monthly_account * (1 - self.private_saving_rate * self.dt)) * self.velocity_of_money * self.dt
        self.gdp_record.append(this_monthly_gdp)


    def t(self):
        time = [None] * len(self.government_spending_record)
        for i in range(0,len(time)):
            time[i] = i * self.dt
        
        return time


    # Iterate
    def step(self):
        self.spend()
        self.tax()
        
        private_sector_monthly_spend = self.money_in_circulation[-1] * self.dt * self.velocity_of_money
        private_sector_net_income    = self.private_sector_balance_record()[-1]        
        private_sector_gross_income  = private_sector_monthly_spend + private_sector_net_income    
        
        self.private_sector_income.append(private_sector_gross_income)          
        self.private_sector_saving.append(private_sector_gross_income * self.private_saving_rate)
        
        new_money = self.money_in_circulation[-1] + private_sector_net_income - self.private_sector_saving[-1]
        self.money_in_circulation.append(new_money)
        


#------------------------------------------------------------

# Initialize object
econ = Economy()

#------------------------------------------------------------

# set up figure and animation
fig = plt.figure()

gdp = fig.add_subplot(111, xlim=(1, 100), ylim=(0, 0.5))
gdp.grid()
gdp_line, = gdp.plot([], [], 'o-', lw=3) # 

gov_spending = fig.add_subplot(111, xlim=(1, 100), ylim=(0, 0.5))
gov_spending.grid()
gov_spending_line, = gdp.plot([], [], 'o-', lw=3) # 


def init():
    # Clear frame on each interation
    gdp_line.set_data([], [])
    gov_spending_line.set_data([], [])
    return gdp_line,gov_spending_line,

def animate(i):
    global econ       
    econ.step()
    
    gdp_line.set_data(econ.t(), econ.private_sector_income)
    gov_spending_line.set_data(econ.t(), econ.government_spending_record)
    return gdp_line, gov_spending_line,

ani = animation.FuncAnimation(fig, animate, frames=500,
                              interval=50, blit=True, init_func=init)

plt.show()