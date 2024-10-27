import { FC, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useNavigate, Outlet } from 'react-router-dom'
import { ConfigProvider } from 'antd'
import Content from '../Content/Content'
// import UploadData from '../UploadData/UploadData'
import styles from './App.module.scss'

const App: FC = () => {
  const navigate = useNavigate()

  useEffect(() => {
    navigate('/upload')
  }, [])

  return (
    <div className={styles.app}>
      <ConfigProvider
        theme={{
          token: {
            fontFamily: 'Roboto',
          },
        }}
      >
        <header className={styles.header}>
          <Link to='/upload' reloadDocument className={styles.logo}>
            <div className={styles.logo} />
          </Link>
        </header>
        <main className={styles.main}>
          <Content>
            <Outlet />
          </Content>
        </main>
        <footer className={styles.footer}>
          <nav>
            <ul className={styles.navigation}>
              <li>Клиентам</li>
              <li>Общая информация</li>
              <li>О фонде</li>
              <li>Раскрытие информации</li>
            </ul>
          </nav>
        </footer>
      </ConfigProvider>
    </div>
  )
}

export default App
