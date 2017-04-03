__author__ = 'andompesta'
import quandl
import configparser
import os



#company = {'apple':'WIKI/AAPL', 'google':'WIKI/GOOGL', 'yahoo':'WIKI/YHOO', 'goldman':'WIKI/GS'}


company = {'apple':'WIKI/AAPL', 'google':'WIKI/GOOGL', 'yahoo':'WIKI/YHOO', 'goldman':'WIKI/GS', 'capital_city_bank':'WIKI/CCBG', 'bank_of_america':'WIKI/BAC', 'sunTrust_banks':'WIKI/STI', 'cantel_medical_corp':'WIKI/CMN', 'ICU_medical':'WIKI/ICUI', 'wright_medical_group':'WIKI/WMGI'}

prop = configparser.ConfigParser()
prop.read('../conf.ini')

collapse = prop.get('ALL', 'collapse')

for company_name in company:
    end_time = prop.get('DATE', 'end_time')
    data = quandl.get(company[company_name], authtoken="zTEsWpGga_5eqG6YCkRS", start_date="2004-10-01", end_date="2017-02-01", collapse=collapse)

    for key in data.keys():
        if '. ' in key:
            data[key.replace('. ', '_')] = data.pop(key)

    to_del = []
    for (low_time, low_value), (high_time, high_value) in zip(data['Low'].iteritems(), data['High'].iteritems()):
        if low_value == high_value:
            assert low_time == high_time
            to_del.append(low_time)
            print('deleted element %s\t%f\t%f\t%s' % (low_time, high_value, low_value, company_name))


    for del_time in to_del:
        data = data.drop(del_time)


    full_path = os.path.join('../data/stock', company_name) + '.csv'
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    data.to_csv(full_path)
