import os
import argparse
from openai import OpenAI
import base64


def encode_image(image_path):
    """将图片转换为Base64编码
    
    Args:
        image_path: 图片文件路径
    
    Returns:
        str: Base64编码的图片字符串
    
    Raises:
        FileNotFoundError: 当图片文件不存在时
        IOError: 当读取图片文件失败时
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    except IOError as e:
        raise IOError(f"读取图片文件失败: {str(e)}")


def create_openai_client():
    """创建OpenAI客户端实例
    
    Returns:
        OpenAI: 配置好的OpenAI客户端实例
    
    Raises:
        ValueError: 当环境变量中没有API密钥时
    """
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        raise ValueError("环境变量ARK_API_KEY未设置，请确保已正确配置API密钥")
    
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key,
    )


def analyze_image(client, image_path, prompt="图片中体现出了用户什么立场？"):
    """分析图片并返回AI的理解结果
    
    Args:
        client: OpenAI客户端实例
        image_path: 图片文件路径
        prompt: 询问图片的提示文本
    
    Returns:
        str: AI对图片的分析结果
    
    Raises:
        Exception: 当API调用失败时
    """
    # 编码图片
    base64_image = encode_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model="doubao-seed-1-6-vision-250815",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"API调用失败: {str(e)}")


def main():
    """主函数，处理命令行参数并执行图片分析"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用AI分析图片内容")
    parser.add_argument("--image", type=str, default="测试.png", 
                        help="图片文件路径 (默认: 测试.png)")
    parser.add_argument("--prompt", type=str, 
                        default="图片中体现出了用户什么立场？",
                        help="分析图片的提示文本")
    args = parser.parse_args()
    
    try:
        # 创建客户端
        client = create_openai_client()
        
        # 分析图片
        result = analyze_image(client, args.image, args.prompt)
        
        # 输出结果
        print("\n图片分析结果:")
        print("=" * 50)
        print(result)
        print("=" * 50)
        
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
