from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, TimeoutException
import pandas as pd
from pathlib import Path
import time

def retry_on_stale_element(max_attempts=3, delay=1):
    """处理 StaleElementReferenceException 的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except StaleElementReferenceException:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_on_stale_element(max_attempts=3)
def get_policy_detail(driver, url):
    """获取政策详细信息"""
    try:
        driver.get(url)
        time.sleep(2)  # 等待页面加载
        
        # 等待表格加载，添加更长的超时时间
        wait = WebDriverWait(driver, 15)
        table = wait.until(
            EC.presence_of_element_located((By.CLASS_NAME, "xxgk"))
        )
        
        # 确保页面完全加载
        wait.until(
            EC.visibility_of_element_located((By.CLASS_NAME, "xxgk"))
        )
        
        policy_detail = {}
        
        try:
            # 直接使用 XPath 定位主题分类
            th_element = driver.find_element(By.XPATH, "//th[contains(text(), '主题分类')]")
            td_element = th_element.find_element(By.XPATH, "./following-sibling::td")
            policy_detail['主题分类'] = td_element.text.strip()
        except Exception as e:
            print(f"未找到主题分类信息: {e}")
            policy_detail['主题分类'] = ""
        
        return policy_detail
    
    except Exception as e:
        print(f"获取政策详情失败: {e}")
        return {'主题分类': ""}

def get_policy_list(driver, page_url):
    """使用Selenium获取单页政策列表"""
    max_retries = 3
    for retry in range(max_retries):
        try:
            # 访问页面
            driver.get(page_url)
            time.sleep(5)  # 等待页面加载
            
            # 等待内容加载
            wait = WebDriverWait(driver, 15)
            
            # 确保页面框架加载完成
            wait.until(
                EC.presence_of_element_located((By.CLASS_NAME, "simple_pgContainer"))
            )
            
            page_policies = []
            
            # 使用JavaScript获取所有政策项
            items = driver.execute_script("""
                return document.getElementsByClassName('xzgfx_list_item');
            """)
            
            print(f"找到 {len(items)} 条政策")
            
            for item in items:
                try:
                    policy = {}
                    
                    # 使用JavaScript获取元素文本和属性
                    title_info = driver.execute_script("""
                        var item = arguments[0];
                        var titleElement = item.querySelector('.xzgfx_list_title2 a');
                        return {
                            title: titleElement ? titleElement.getAttribute('title') : '',
                            href: titleElement ? titleElement.getAttribute('href') : '',
                            text: titleElement ? titleElement.textContent : ''
                        };
                    """, item)
                    
                    policy['文件名称'] = title_info['title'] or title_info['text']
                    policy['来源'] = title_info['href']
                    policy['本地文件名'] = "".join(char for char in policy['文件名称'] if char.isalnum() or char.isspace())
                    
                    # 获取其他信息
                    other_info = driver.execute_script("""
                        var item = arguments[0];
                        return {
                            docNumber: item.querySelector('.xzgfx_list_title3') ? item.querySelector('.xzgfx_list_title3').textContent : '',
                            pubDate: item.querySelector('.xzgfx_list_title4') ? item.querySelector('.xzgfx_list_title4').textContent : '',
                            effectDate: item.querySelector('.xzgfx_list_title5') ? item.querySelector('.xzgfx_list_title5').textContent : ''
                        };
                    """, item)
                    
                    policy['文号'] = other_info['docNumber']
                    policy['成文日期'] = other_info['pubDate']
                    policy['发布日期'] = other_info['effectDate']
                    
                    # 获取详细页面信息
                    if policy['来源']:
                        try:
                            # 打开新标签页
                            driver.execute_script(f"window.open('{policy['来源']}', '_blank');")
                            time.sleep(2)
                            
                            # 切换到新标签页
                            driver.switch_to.window(driver.window_handles[-1])
                            
                            # 等待详细信息加载
                            wait = WebDriverWait(driver, 10)
                            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "xxgk")))
                            
                            # 获取主题分类
                            theme = driver.execute_script("""
                                var thElement = document.evaluate("//th[contains(text(), '主题分类')]", 
                                    document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                                if (thElement) {
                                    var tdElement = thElement.nextElementSibling;
                                    return tdElement ? tdElement.textContent : '';
                                }
                                return '';
                            """)
                            
                            policy['主题分类'] = theme.strip() if theme else ""
                            
                            # 关闭详细页面标签
                            driver.close()
                            
                            # 切回主页面
                            driver.switch_to.window(driver.window_handles[0])
                            
                        except Exception as e:
                            print(f"获取详细信息失败: {e}")
                            policy['主题分类'] = ""
                            # 确保切回主页面
                            if len(driver.window_handles) > 1:
                                driver.close()
                            driver.switch_to.window(driver.window_handles[0])
                    
                    page_policies.append(policy)
                    print(f"成功获取政策: {policy['文件名称']}")
                    
                except Exception as e:
                    print(f"处理单条政策时出错: {e}")
                    continue
            
            if page_policies:
                return page_policies
            
            print(f"第 {retry + 1} 次尝试未获取到数据，重试...")
            time.sleep(3)
            
        except Exception as e:
            print(f"获取政策列表失败 (尝试 {retry + 1}/{max_retries}): {e}")
            if retry < max_retries - 1:
                time.sleep(3)
                continue
            return []
    
    return []

def save_to_csv(policies, output_file):
    """将政策信息保存到CSV文件"""
    try:
        if not policies:
            print("没有数据可保存")
            return
            
        # 创建DataFrame
        df = pd.DataFrame(policies)
        
        # 确保输出目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为CSV，使用utf-8-sig编码以支持中文
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"成功保存到文件: {output_file}")
        
        # 打印统计信息
        print("\n数据统计:")
        print(f"总条数: {len(df)}")
        if '主题分类' in df.columns:
            print("\n主题分类分布:")
            print(df['主题分类'].value_counts())
        print("\n文号分布:")
        print(df['文号'].value_counts())
        
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def main():
    base_url_template = 'https://www.zj.gov.cn/col/col1229697834/index.html?number=C0103&pageNum={}'
    total_pages = 136
    all_policies = []
    current_page = 1
    
    try:
        # 初始化Chrome驱动
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        while current_page <= total_pages:
            print(f"\n正在获取第 {current_page} 页...")
            page_url = base_url_template.format(current_page)
            
            # 获取当前页的政策列表
            page_policies = get_policy_list(driver, page_url)
            
            if page_policies:
                all_policies.extend(page_policies)
                print(f"第 {current_page} 页获取完成，当前共获取 {len(all_policies)} 条政策")
                
                # 每获取5页保存一次
                if current_page % 5 == 0:
                    save_to_csv(all_policies, f'zjpolices_temp_{current_page}.csv')
                    print(f"已完成 {current_page} 页的数据获取，临时保存完成")
                
                current_page += 1
            else:
                print(f"第 {current_page} 页获取失败，重试...")
                time.sleep(5)  # 失败后等待更长时间
                continue
            
            time.sleep(3)  # 页面间隔
            
    except Exception as e:
        print(f"获取过程中出错: {e}")
    
    finally:
        try:
            driver.quit()
        except:
            pass
        
        if all_policies:
            save_to_csv(all_policies, 'zjpolices_final.csv')
            print(f"\n总共获取到 {len(all_policies)} 条政策信息")
        else:
            print("未获取到任何政策信息")

if __name__ == "__main__":
    main() 
