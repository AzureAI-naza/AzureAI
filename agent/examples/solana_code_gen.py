import requests
import json

# Use HTTP request to verify Solana web3.js code
def test_code(code: str):
    url = "http://localhost:8081/api/v1/verify/solana/code"
    payload = {"code": code}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"Request successful, code:\n{code}\nResponse:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # 打印验证结果的解析
        if result.get("status") == "success" and "results" in result:
            verification = result["results"].get("verification", "Unknown")
            knowledge_count = result["results"].get("knowledge_retrieved", 0)
            print(f"\nRetrieved {knowledge_count} related knowledge items")
            print(f"Verification result: {verification}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response:
            print(e.response.text)
    except Exception as e:
        print(f"Verification failed: {e}")


def main():
    samples = [
        ("Valid Solana web3.js code", "const connection = new Connection('https://api.mainnet-beta.solana.com');\nconst balance = await connection.getBalance(publicKey);"),
        ("Incorrect API usage", "const connection = new Connection('https://api.mainnet-beta.solana.com');\nconst balance = connection.getAccountBalance(publicKey);"),
        ("Non-existent API", "const connection = new Connection('https://api.mainnet-beta.solana.com');\nconst nft = await connection.getNFTMetadata(tokenAddress);"),
        ("Non-code input", "This is not Solana web3.js code")
    ]
    
    for title, code in samples:
        print(f"=== {title} ===")
        test_code(code)
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
