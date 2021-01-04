#!/usr/bin/env python
# coding: utf-8

# In[119]:


import requests
from bs4 import BeautifulSoup as bs
import csv
import re


# In[155]:


def get_page(url):
    response = requests.get(url)
    
    if not response.ok:
        print('Server Responded: ', response.status_code)
    else:
        soup = bs(response.text, 'lxml')
    return soup


def get_data(soup):
    try:
        title = soup.find('h1', class_='text-bold text-size-400 text-size-sm-700').get_text().strip()
    except:
        title = ''
        
    try:
        price = soup.find('span', class_='first-price').get_text()
    except:
        price = ''
        
    try:
        specs = soup.find_all('div', class_='col-xs-8')
        specs_list = [item.get_text() for item in specs]
        spec_names = ['Mileage', 'Drive Type', 'Engine', 'Transmission', 'Fuel Type', 'MPG', 'Exterior', 'Interior', 'VIN']
        combined_specs = zip(spec_names, specs_list)
        specs_dict = dict(combined_specs)
    except:
        specs = ''
        
    data_dict = {
        'Title': title,
        'Price': price,
    }
    
    data_dict.update(specs_dict)
    
    return data_dict


def get_index_data(soup):
    try:
        links = soup.find_all('a', attrs={'rel':'nofollow'})
    except:
        links = []
    
    urls = [link['href'] for link in links]
    actual = [url for url in urls if not url.startswith('tel')]
    full_urls = [f'https://www.kbb.com{url}' for url in actual][::2]
    
    return full_urls


# In[183]:


def write_csv(data):
    with open('kbb_scraper.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        
        try:
            row = [data['Title'], data['Price'], data['Mileage'], data['Drive Type'], data['Engine'],
                  data['Transmission'], data['Fuel Type'], data['MPG'], data['Exterior'], data['Interior']]
            
            writer.writerow(row)
        except:
            try: 
                row = [data['Title'], data['Price'], data['Mileage'], data['Drive Type'], data['Engine'],
                data['Transmission'], data['Fuel Type'], data['MPG'], data['Exterior']]
                
                writer.writerow(row)
            except:
                try:
                    row = [data['Title'], data['Price'], data['Mileage'], data['Drive Type'], data['Engine'],
                    data['Transmission'], data['Fuel Type'], data['MPG']]
            
                    writer.writerow(row)
                except:
                    row = [data['Title']]
                    writer.writerow(row)


# In[171]:


url = 'https://www.kbb.com/cars-for-sale/all/?distance=75'
get_index_data(get_page(url))

url_records = [f'https://www.kbb.com/cars-for-sale/all/columbia-mo-65201?distance=75&dma=&channel=KBB&searchRadius=75&isNewSearch=false&marketExtension=include&showAccelerateBanner=false&sortBy=relevance&numRecords=25&firstRecord={x}' 
              for x in range(1000) if x % 25 == 0]


# In[185]:


url_records_test = url_records[::50]

for url in url_records:
    get_data(get_page(url))
    car_urls = get_index_data(get_page(url))

    for idx, link in enumerate(car_urls):
        data = get_data(get_page(link))
        write_csv(data)
    
        if idx % 25 == 0:
            print(f'{idx} iteration complete')
            print(data)
            print('-----------------------------------------------------------')


# In[186]:


get_ipython().system('jupyter nbconvert --to script kbb-scraper.ipynb')


# In[ ]:




