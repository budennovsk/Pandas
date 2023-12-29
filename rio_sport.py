#
# from selenium import webdriver
# from fake_useragent import UserAgent
# from selenium.webdriver.chrome.options import Options
# import time
# from selenium.webdriver.common.by import By
# from tqdm import tqdm
#
#
# for _ in tqdm(range(100)):
#     # Создание экземпляра UserAgent
#     ua = UserAgent()
#     user_agent = ua.random
#     # Опции браузера
#     options = Options()
#     # options.add_argument("--headless")  # Включение headless режима
#     options.add_argument(f'user-agent={user_agent}')
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_experimental_option("useAutomationExtension", False)
#     options.add_experimental_option("excludeSwitches",["enable-automation"])
#     # options.add_argument("--headless")
#     # Путь к драйверу браузера (в данном случае Chrome)
#     driver_path = 'chromeexe/chromedriver.exe'
#     # Создание экземпляра драйвера
#     driver = webdriver.Chrome()
#     # url = 'https://intoli.com/blog/not-possible-to-block-chrome-headless/chrome-headless-test.html'
#     # driver.get(url)
#     # time.sleep(10)
#
#
#     # Открытие ссылки
#     url = 'https://rsport.ria.ru/20231219/sportswoman-of-year-1909228789.html'
#     driver.get(url)
#     time.sleep(2)
#     element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[4]/div[2]/div[7]/div[1]/div[1]')
#     time.sleep(2)
#     element.click()
#     time.sleep(2)
#     element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[5]/div/div/div[2]/div/div[2]/div/div[8]/div/div/div[2]/div[1]/div[2]/div')
#     time.sleep(2)
#     element.click()
#     time.sleep(2)
#     element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[6]/div/div/div[1]/div')
#     time.sleep(2)
#     element.click()
#     time.sleep(2)
#
#     # Закрытие драйвера
#     driver.quit()



#
# from selenium import webdriver
# from fake_useragent import UserAgent
# import time
# from selenium.webdriver.common.by import By
# import undetected_chromedriver as UC
#
# # Создание экземпляра UserAgent
# ua = UserAgent()
# user_agent = ua.random
#
# # Опции браузера
# options = webdriver.ChromeOptions()
# options.add_argument(f'user-agent={user_agent}')
# options.add_argument("--disable-blink-features=AutomationControlled")
# options.add_argument('start-maximized')
# options.add_argument('disable-infobars')
# # Другие опции браузера...
#
# # Путь к драйверу браузера (в данном случае Chrome)
# driver_path = 'chromeexe/chromedriver.exe'
#
# # Создание экземпляра драйвера
# # driver = webdriver.Chrome(options=options)
# driver = UC.Chrome()
# # Открытие ссылки
# # url = 'https://www.whatismybrowser.com/detect/what-is-my-user-agent'
# url='https://intoli.com/blog/not-possible-to-block-chrome-headless/chrome-headless-test.html'
# driver.get(url)
# time.sleep(100)
#
#
#
# # # Вывод User-Agent
# # user_agent_element = driver.find_element(By.CSS_SELECTOR, '.string-major')
# # print('User-Agent:', user_agent_element.text)
#
# # Закрытие драйвера
# driver.quit()



from selenium import webdriver
from fake_useragent import UserAgent
from selenium.webdriver.chrome.options import Options
import time
from selenium.webdriver.common.by import By
from tqdm import tqdm


for _ in tqdm(range(100)):
    # Создание экземпляра UserAgent
    ua = UserAgent()
    user_agent = ua.random
    # Опции браузера
    options = Options()
    # options.add_argument("--headless")  # Включение headless режима
    options.add_argument(f'user-agent={user_agent}')
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("useAutomationExtension", False)
    options.add_experimental_option("excludeSwitches",["enable-automation"])
    # options.add_argument("--headless")
    # Путь к драйверу браузера (в данном случае Chrome)
    driver_path = 'chromeexe/chromedriver.exe'
    # Создание экземпляра драйвера
    driver = webdriver.Chrome(options=options)
    # url = 'https://intoli.com/blog/not-possible-to-block-chrome-headless/chrome-headless-test.html'
    # driver.get(url)
    # time.sleep(10)


    # Открытие ссылки
    url = 'https://rsport.ria.ru/20231219/sportswoman-of-year-1909228789.html'
    driver.get(url)
    time.sleep(5)
    element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[4]/div[2]/div[8]/div[1]/div[1]/img')
    time.sleep(5)
    element.click()
    time.sleep(5)
    element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[5]/div/div/div[2]/div/div[2]/div/div[9]/div/div/div[2]/div[1]/div[2]/div')
    time.sleep(5)
    element.click()
    time.sleep(5)
    element = driver.find_element(By.XPATH, '//*[@id="endless"]/div[1]/div/div/div/div[1]/div[1]/div/div[1]/div[1]/div/div/div/div[6]/div/div/div[1]/div')
    time.sleep(5)
    element.click()
    time.sleep(5)

    # Закрытие драйвера
    driver.quit()
