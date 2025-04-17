/**
 * DyeOptimAI 결과 분석기 스크립트
 * 의류 염색 공정 최적화 AI 시스템의 결과 파일 분석 및 해석을 위한 JavaScript
 */

// 페이지 로드 시 실행
document.addEventListener('DOMContentLoaded', function() {
    // 기본 탭 활성화
    document.querySelector('.tab-button').click();

    // 이미지 확대 기능
    const images = document.querySelectorAll('.image-container img');
    images.forEach(img => {
        img.addEventListener('click', function() {
            if (this.classList.contains('enlarged')) {
                this.classList.remove('enlarged');
                this.style.transform = '';
                this.style.cursor = 'zoom-in';
                this.style.zIndex = '';
            } else {
                this.classList.add('enlarged');
                this.style.transform = 'scale(1.5)';
                this.style.cursor = 'zoom-out';
                this.style.zIndex = '1000';
            }
        });

        img.style.cursor = 'zoom-in';
    });
});

/**
 * 탭 전환 함수
 * @param {Event} evt - 이벤트 객체
 * @param {string} tabName - 활성화할 탭 ID
 */
function openTab(evt, tabName) {
    // 모든 탭 컨텐츠 숨기기
    var tabContents = document.getElementsByClassName("tab-content");
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove("active");
    }

    // 모든 탭 버튼 비활성화
    var tabButtons = document.getElementsByClassName("tab-button");
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove("active");
    }

    // 클릭된 탭 활성화
    document.getElementById(tabName).classList.add("active");
    evt.currentTarget.classList.add("active");

    // URL 해시 업데이트 (페이지 이동 없이)
    history.pushState(null, null, '#' + tabName);
}

/**
 * 코드 블록에 구문 강조 적용
 * 주의: 단순한 정규식 기반 구문 강조는 HTML 구문과 충돌할 수 있습니다.
 * 실제 구현에서는 highlight.js와 같은 라이브러리 사용을 권장합니다.
 */
function highlightCode() {
    // 코드 강조 기능 비활성화 - HTML과 충돌 문제 방지
    // 실제 구현에서는 highlight.js 같은 라이브러리를 사용하세요
    console.log("코드 강조 기능이 비활성화 되었습니다.");

    // 코드 블록에 pre 스타일 적용 (들여쓰기와 개행 유지)
    const codeBlocks = document.querySelectorAll('.code-box');
    codeBlocks.forEach(block => {
        block.style.whiteSpace = 'pre';
    });
}

// 페이지 로드 및 해시 변경 시 처리
window.addEventListener('hashchange', handleHashChange);
window.addEventListener('load', handleHashChange);

/**
 * 코드 블록에 구문 강조 적용 (단순 구현)
 */
function highlightCode() {
    const codeBlocks = document.querySelectorAll('.code-box');
    codeBlocks.forEach(block => {
        let content = block.innerHTML;

        // 키워드 강조
        const keywords = ['function', 'return', 'if', 'else', 'for', 'while', 'var', 'let', 'const'];
        keywords.forEach(keyword => {
            const regex = new RegExp(`\\b${keyword}\\b`, 'g');
            content = content.replace(regex, `<span style="color: #569CD6;">${keyword}</span>`);
        });

        // 문자열 강조
        content = content.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g,
            match => `<span style="color: #CE9178;">${match}</span>`);

        // 주석 강조
        content = content.replace(/\/\/.*$/gm,
            match => `<span style="color: #6A9955;">${match}</span>`);

        block.innerHTML = content;
    });
}

// 페이지 로드 후 코드 강조 실행
window.addEventListener('load', highlightCode);