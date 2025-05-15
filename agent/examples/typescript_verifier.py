import requests
import json

# Make real requests to the running FastAPI service for TypeScript verification

def test_code(code: str):
    url = "http://localhost:8081/api/v1/verify/typescript"
    payload = {"code": code}
    response = requests.post(url, json=payload)
    try:
        response.raise_for_status()
        print(f"Request successful, code:\n{code}\nResponse:")
        print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Request failed: {e}")
        print(response.text)


def main():
    samples = [
        ("Valid TypeScript code", "function add(a: number, b: number): number { return a + b; }"),
        ("Type error code", "const x: string = 123;"),
        ("Non-code input", "This is not TypeScript code")
    ]
    for title, code in samples:
        print(f"=== {title} ===")
        test_code(code)
        print()


if __name__ == "__main__":
    main()
