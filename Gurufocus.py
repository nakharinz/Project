from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# Navigate to the website
url = "https://www.gurufocus.com/stock/LULU/summary"
driver.get(url)

# Wait for the content to load (you can use explicit or implicit waits)
driver.implicitly_wait(10)  # wait for 10 seconds

# Locate the element using XPath
xpath = '//*[@id="components-root"]/div[1]/div[4]/div[2]/div[2]/div[1]/div[1]/div/span/div/div'
element = driver.find_element(By.XPATH, xpath)

# Extract the content
content = element.text

# Print the content
print(content)

# Close the WebDriver
driver.quit()
