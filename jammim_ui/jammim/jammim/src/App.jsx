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
  const recognitionRef = useRef(null)


  // 컴포넌트 unmount 시 인식 종료(clean-up)
  useEffect(() => {
    return () => {
      if (recognitionRef.current) {
        recognitionRef.current.abort();
        recognitionRef.current = null;
      }
    }
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
  </div>
 );


}
export default App;
