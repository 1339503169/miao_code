from selenium import webdriver
import time
import json

from selenium.webdriver.chrome.options import Options

def data_to_json(dict):
    with open('data/administrative_sanction_anhui_feidong.json','a',encoding='utf-8') as f:
        json.dump(dict,f)
        f.close()
def get_administrative_sanction(url):
    options = Options()
    options.headless = True
    browser = webdriver.Chrome(options=options)
    browser.get(url)
    time.sleep(1)
    page = 1
    while True:
        print('开始爬第{}页'.format(page))
        try:
            a = browser.find_elements_by_class_name('btn-detail')
            summary = []
            for i in a:
                i.click()
                time.sleep(0.5)
                handles = browser.window_handles
                browser.switch_to.window(handles[1])
                if len(browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[1]/td[2]').text) > 3:
                    url = browser.current_url
                    org = browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[1]/td[2]').text
                    code = browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[3]/td[2]').text
                    result = browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[19]/td[2]').text
                    data = browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[23]/td[2]').text
                    gover = browser.find_element_by_xpath('//*[@id="mainTable"]/tbody/tr[26]/td[2]').text
                    summary.append([url, org, code, result, data, gover])
                browser.close()
                browser.switch_to.window(handles[0])
            if len(summary)>0:
                data_to_json(summary)
        except Exception as e:
            print(e)
            pass
        summary = []
        browser.find_element_by_class_name('next').click()
        time.sleep(1)
        print('第{}页完成'.format(page))
        page+=1

if __name__=='__main__':
    url='http://117.66.242.222:8800/credit-website/publicity/public/punishment-list.do?navId=250AF3641A5D19C6E05010ACD33A5781&columnId=272953CCCF2901D5E05010ACD33A4F3E'
    get_administrative_sanction(url)










