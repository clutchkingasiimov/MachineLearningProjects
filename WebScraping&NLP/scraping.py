from bs4 import BeautifulSoup
import requests

"""
Current action plan:

1. Devise a way to source in links and store them in a list
2. Clean the whitespace from the product name tags
3. Filter the product name, brand name, cost and its reviews
4. Try to determine the average rating of the product (mean, variance) [Numpy]
5. Store the text information in a dataframe [Use pandas for this]
6. Forward feed processed data in NN for sentiment polarity (-1 < x < 1) [Use textblob for this]
7. Predictive algorithm for predicting future sentiment trends once 
features have been labeled with polarities (Logistic regression, Gaussian NB, Random Forests) [H20/Sklearn]
"""

#Pull html from the link/links (store the information in a list)
links = ['https://www.amazon.in/dp/B00NBILSD6/ref=br_msw_pdt-2?_encoding=UTF8&smid=AH2BV6QWKVD69&pf_rd_m=A1VBAL9TL5WCBF&pf_rd_s=&pf_rd_r=WWRDKXE1GYFPMKGYZFX0&pf_rd_t=36701&pf_rd_p=2b9bb3c1-71bb-48bb-8476-31c6e37895b1&pf_rd_i=desktop',
'https://www.amazon.in/dp/B07DPQRCQB/ref=sspa_dk_detail_0?psc=1&pd_rd_i=B07DPQRCQB&pd_rd_w=rLYi2&pf_rd_p=02c0700e-cd85-479f-b670-81a01283e38b&pd_rd_wg=x2VlC&pf_rd_r=B696KN9EKDWFA365KYF0&pd_rd_r=1b8b2ce7-6cef-11e9-9965-c7ce4efafbad']

names = [] #Empty list to store product names 
for link in links:
    url = requests.get(link)
    soup = BeautifulSoup(url.content, 'html.parser')
    product = soup.find("h1", class_="a-size-large a-spacing-none") #Needs a tag and its class to access string attr.
    product_name = product.span.text #Access text from the child tag "span" using the parent tag "h1". 
    names.append(product_name)
    
names = [element.strip() for element in names] #Strip whitespace from the list and return names of the product
print(names)

#Each string at the end has the color of the product, which will carry on for other types of products also. Remove them

"""
url = requests.get('https://www.amazon.in/dp/B00NBILSD6/ref=br_msw_pdt-2?_encoding=UTF8&smid=AH2BV6QWKVD69&pf_rd_m=A1VBAL9TL5WCBF&pf_rd_s=&pf_rd_r=WWRDKXE1GYFPMKGYZFX0&pf_rd_t=36701&pf_rd_p=2b9bb3c1-71bb-48bb-8476-31c6e37895b1&pf_rd_i=desktop')
print(url)
#Parse the HTML and instantiate to variable 'soup'
soup = BeautifulSoup(url.content, 'html.parser')
#print(soup.prettify()) #Shows the HTML of the code 

#Title of the html page
product = soup.find("h1", class_="a-size-large a-spacing-none")
print(product)
#Finds the header text 
"""
