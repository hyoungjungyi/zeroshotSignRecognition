import { useState, useRef, useEffect } from 'react';
import './App.css';
import "./index.css";
import ArcReactorEffect ,{SpinningArc} from './components/ArcReactorEffect';
import GesturePanel from './components/GesturePanel';
import ClockWeatherPanel from './components/ClockWeatherPanel';

function App() {
  const [count, setCount] = useState(0)
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [llmText, setLlmText] = useState('');
  const recognitionRef = useRef(null)
  const handleLLMText = (msg) => {
    console.log('[Renderer] LLM text update:', msg);
    setLlmText(msg);
  };


  // 컴포넌트 unmount 시 인식 종료(clean-up)
  useEffect(() => {
    let unsubscribe = null;

    // electronAPI 존재 여부 확인 로그
    console.log('[Renderer] window.electronAPI:', window.electronAPI);

    if (window.electronAPI && typeof window.electronAPI.onLLMResponse === 'function') {
      unsubscribe = window.electronAPI.onLLMResponse((msg) => {
        console.log('[Renderer] LLM response received:', msg);
        handleLLMText(msg);
      });
    } else {
      console.warn('[Renderer] electronAPI.onLLMResponse not available');
    }

    // window.postMessage fallback 수신
    const messageHandler = (event) => {
      if (event.data && event.data.type === 'llm-response') {
        console.log('[Renderer] llm-response via postMessage:', event.data.payload);
        handleLLMText(event.data.payload);
      }
    };
    window.addEventListener('message', messageHandler);

    return () => {
      if (typeof unsubscribe === 'function') {
        unsubscribe();
      }
      window.removeEventListener('message', messageHandler);
      if (recognitionRef.current) {
        recognitionRef.current.abort();
        recognitionRef.current = null;
      }
    };
  }, []);

  const handlePythonStart = () => {
    console.log(window.electronAPI);
    console.log(typeof window.electronAPI?.startPython);
    if (window.electronAPI && typeof window.electronAPI.startPython === 'function') {
      window.electronAPI.startPython();
      console.log('Python 실행 요청 IPC 전송 완료');
    } else {
      console.error('electronAPI.startPython이 준비되어 있지 않습니다.');
    }
  };

  return (
  <div
    style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gridTemplateRows: 'auto auto',
      gap: '20px',
      padding: 20,
      color: 'white',
    }}
  >
    <div>
      <GesturePanel />
    </div>
    <div>
      <ArcReactorEffect />
    </div>
    <div>
      <ClockWeatherPanel />
    </div>
    <div style={{ gridColumn: '1 / span 3', textAlign: 'center' }}>
      <button
        onClick={handlePythonStart}
        style={{
          marginTop: 10,
          padding: '10px 20px',
          fontSize: '16px',
          cursor: 'pointer',
        }}
      >
        {`Activate Motion Capture`}
      </button>
    </div>
    <div style={{ gridColumn: '1 / span 3', textAlign: 'center' }}>
      <div
        style={{
          marginTop: 12,
          padding: '16px 20px',
          background: 'rgba(0,0,0,0.7)',
          border: '2px solid #4fc3f7',
          borderRadius: 10,
          minHeight: 60,
          fontSize: 18,
          fontWeight: 700,
        }}
      >
        {llmText || 'LLM 응답을 기다리는 중...'}
      </div>
    </div>
  </div>
 );


}
export default App;
