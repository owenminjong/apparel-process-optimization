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
    // 모든 탭 콘텐츠 숨기기
    var tabContents = document.getElementsByClassName("tab-content");
    for (var i = 0; i < tabContents.length; i++) {
        tabContents[i].style.display = "none";
    }

    // 모든 탭 버튼에서 active 클래스 제거
    var tabButtons = document.getElementsByClassName("tab-button");
    for (var i = 0; i < tabButtons.length; i++) {
        tabButtons[i].className = tabButtons[i].className.replace(" active", "");
    }

    // 클릭된 탭 표시 및 active 클래스 추가
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";

    // URL 해시 업데이트 (페이지 새로고침 없이)
    window.location.hash = tabName;
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

function handleHashChange() {
    var hash = window.location.hash.substring(1);
    if (hash) {
        // 해당 ID를 가진 탭 버튼 찾기
        var buttons = document.getElementsByClassName("tab-button");
        for (var i = 0; i < buttons.length; i++) {
            var button = buttons[i];
            var tabId = button.getAttribute("onclick").match(/openTab\(event, ['"](.+)['"]\)/)[1];
            if (tabId === hash) {
                // 해당 탭 버튼 클릭
                button.click();
                return;
            }
        }
    }

    // 해시가 없거나 일치하는 탭이 없으면 첫 번째 탭 표시
    document.getElementsByClassName("tab-button")[0].click();
}

// 페이지 로드 시 실행
document.addEventListener("DOMContentLoaded", function() {
    // URL 해시에 따라 탭 표시
    handleHashChange();

    // 해시 변경 시 탭 변경
    window.addEventListener("hashchange", handleHashChange);
});