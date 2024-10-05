def apply_markdown_format(original):
    # Split the original text into lines
    lines = original.strip().split('\n')
    
    # Apply Markdown formatting
    formatted_lines = []
    for i, line in enumerate(lines):
        if i == 0:
            # Add Markdown header to the first line
            formatted_lines.append(f"## {line}")
        elif 2 <= i <= 7:
            continue
        else:
            formatted_lines.append(line)

    # Add the Markdown code block syntax
    formatted_lines.insert(3, "```python")
    formatted_lines.append("```")
    
    # Join the formatted lines into a single string
    target = '\n'.join(formatted_lines)
    return target

def test_apply_markdown_format():
    # Original string
    original_text = """문제 20: 텐서의 특정 값 인덱스 찾기
    다음 텐서 x에서 값이 5인 첫 번째 위치의 인덱스를 찾는 코드를 작성하세요:


    python

    Copy code


    x = torch.tensor([3, 5, 7, 5, 9])"""

    # Apply Markdown format
    target_text = apply_markdown_format(original_text)
    print(target_text)

def main():
    test_apply_markdown_format()

if __name__ == '__main__':
    main()