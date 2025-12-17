"""
API 키 확인 스크립트
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 50)
print("API 키 확인")
print("=" * 50)

# OpenAI API 키 확인
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    key_preview = openai_key[:10] + "..." + openai_key[-4:] if len(openai_key) > 14 else "***"
    print(f"✅ OPENAI_API_KEY: {key_preview}")
    
    # OpenAI API 테스트
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        
        # 간단한 API 호출 테스트
        print("\nOpenAI API 연결 테스트 중...")
        response = client.models.list()
        print("✅ OpenAI API 연결 성공!")
        
        # 계정 정보 확인
        print("\n계정 정보 확인 중...")
        try:
            # 사용량 확인 (간단한 모델 리스트로 테스트)
            print(f"✅ 사용 가능한 모델 수: {len(list(response))}")
        except Exception as e:
            print(f"⚠️ 계정 정보 확인 중 오류: {e}")
            
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "429" in error_msg:
            print("\n❌ OpenAI API 할당량 오류!")
            print("   계정에 크레딧이 없거나 할당량을 초과했습니다.")
            print("   https://platform.openai.com/account/billing 에서 확인하세요.")
        elif "Invalid API key" in error_msg or "401" in error_msg:
            print("\n❌ OpenAI API 키가 유효하지 않습니다!")
            print("   .env 파일의 OPENAI_API_KEY를 확인하세요.")
        else:
            print(f"\n❌ OpenAI API 오류: {e}")
else:
    print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
    print("   .env 파일에 OPENAI_API_KEY를 추가하세요.")

# Tavily API 키 확인
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    key_preview = tavily_key[:10] + "..." + tavily_key[-4:] if len(tavily_key) > 14 else "***"
    print(f"\n✅ TAVILY_API_KEY: {key_preview}")
else:
    print("\n⚠️ TAVILY_API_KEY가 설정되지 않았습니다.")
    print("   웹 검색 기능이 작동하지 않습니다.")

print("\n" + "=" * 50)
print("확인 완료")
print("=" * 50)

