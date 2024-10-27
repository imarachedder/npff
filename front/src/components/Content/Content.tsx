import { FC, ReactNode } from 'react'
import styles from './Content.module.scss'

interface ContentProps {
  children: ReactNode
}

const Content: FC<ContentProps> = ({ children }) => {
  return (
    <section className={styles.content}>
      <h1 className={styles.title}>Прогнозирование раннего выхода на пенсию</h1>
      {children}
    </section>
  )
}

export default Content
