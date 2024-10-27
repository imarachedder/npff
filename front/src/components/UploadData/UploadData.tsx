import { CSSProperties, FC, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { FileOutlined } from '@ant-design/icons'
import axios from 'axios'
import { Button, message, Upload } from 'antd'
import type { GetProp, UploadFile, UploadProps } from 'antd'
import styles from './UploadData.module.scss'

const chooseButtonStyle: CSSProperties = {
  width: 284,
  height: 67,
  fontSize: 20,
  borderRadius: 10,
  color: '#0489FA',
  borderColor: '#0489FA',
  backgroundColor: 'transparent',
  marginBottom: 20,
}

const uploadButtonStyle: CSSProperties = {
  borderRadius: 10,
  // backgroundColor: '#0489FA',
  marginTop: 30,
}

type FileType = Parameters<GetProp<UploadProps, 'beforeUpload'>>[0]

const UploadData: FC = () => {
  const navigate = useNavigate()
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [uploading, setUploading] = useState(false)

  const handleUpload = () => {
    const formData = new FormData()
    fileList.forEach((file, index) => {
      if (index === 0) {
        formData.append('contributers', file as FileType)
        console.log('! ', formData.entries)
      } else {
        formData.append('transactions', file as FileType)
        console.log('!! ', formData.entries)
      }
    })
    setUploading(true)
    // const boundary =
    //   '----CustomBoundary' + Math.random().toString(36).substring(2, 15)

    // fetch('http://v2016237.hosted-by-vdsina.ru/predict/', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': `multipart/form-data`,
    //   },
    //   body: formData,
    // })
    //   .then(res => res.json())
    //   .then(res => {
    //     setData(res)
    //     setFileList([])
    //     message.success('Данные успешно отправлены на сервер')
    //     // navigate('/data')
    //   })
    //   .catch(() => {
    //     message.error('Не удалось отправить данные на сервер')
    //   })
    //   .finally(() => {
    //     setUploading(false)
    //     navigate('/data')
    //   })

    // return axios({
    //   method: 'POST',
    //   url: 'http://v2016237.hosted-by-vdsina.ru/predict/',
    //   data: formData,
    //   headers: {
    //     'Content-Type': `multipart/form-data`,
    //   },
    // })

    axios
      .post('http://v2016237.hosted-by-vdsina.ru/predict/', formData, {
        headers: {
          // Accept: '*/*',
          // Connection: 'keep-alive',
          'Accept-Encoding': 'gzip, deflate, br',
          'Content-Type':
            'multipart/form-data; boundary=--------------------------509189379227855950813333',
          'Content-Length': '3154',
        },
      })
      .then(res => {
        setFileList([])
        message.success('Данные успешно отправлены на сервер')
        navigate('/data')
        console.log('res: ', res)
      })
      .catch(error => {
        message.error('Не удалось отправить данные на сервер')
        console.log('error: ', error)
      })
      .finally(() => {
        console.log('!!!')
        setUploading(false)
        navigate('/data')
      })
  }

  const handleChange: UploadProps['onChange'] = info => {
    let newFileList = [...info.fileList]

    newFileList = newFileList.slice(-2)
    setFileList(newFileList)
  }

  const props: UploadProps = {
    onChange: handleChange,
    onRemove: file => {
      const index = fileList.indexOf(file)
      const newFileList = fileList.slice()
      newFileList.splice(index, 1)
      setFileList(newFileList)
    },
    listType: 'picture',
    beforeUpload: file => {
      setFileList([...fileList, file])

      const isCSV = file.type === 'text/csv'
      if (!isCSV) {
        message.error(`${file.name} файл должен иметь формат .csv`)
      }
      // return isCSV || Upload.LIST_IGNORE
      return false
    },
    fileList,
  }

  return (
    <div className={styles.upload}>
      <p className={styles.text}>
        Датасет необходимо загружать в формате .csv (2 файла)
      </p>
      <Upload {...props}>
        <Button icon={<FileOutlined />} style={chooseButtonStyle}>
          Выбрать датасет
        </Button>
      </Upload>
      <div className={styles.row}>
        {fileList.length > 0 && (
          <Button
            type='primary'
            onClick={handleUpload}
            disabled={fileList.length !== 2}
            loading={uploading}
            style={uploadButtonStyle}
          >
            Загрузить датасет
          </Button>
        )}
        {fileList.length === 1 && (
          <p className={styles.hint}>Осталось выборать ещё один датасет</p>
        )}
      </div>
    </div>
  )
}

export default UploadData
