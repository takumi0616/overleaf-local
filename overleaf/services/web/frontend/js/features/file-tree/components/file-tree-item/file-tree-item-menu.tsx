import { useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useFileTreeMainContext } from '../../contexts/file-tree-main'
import MaterialIcon from '@/shared/components/material-icon'

function FileTreeItemMenu({ id, name }: { id: string; name: string }) {
  const { t } = useTranslation()
  const { contextMenuCoords, setContextMenuCoords } = useFileTreeMainContext()
  const menuButtonRef = useRef<HTMLButtonElement>(null)

  const isMenuOpen = Boolean(contextMenuCoords)

  function handleClick(event: React.MouseEvent) {
    event.stopPropagation()
    if (!contextMenuCoords && menuButtonRef.current) {
      const target = menuButtonRef.current.getBoundingClientRect()
      setContextMenuCoords({
        top: target.top + target.height / 2,
        left: target.right,
      })
    } else {
      setContextMenuCoords(null)
    }
  }

  return (
    <div className="menu-button btn-group">
      <button
        className="entity-menu-toggle btn btn-sm"
        id={`menu-button-${id}`}
        onClick={handleClick}
        ref={menuButtonRef}
        aria-haspopup="true"
        aria-expanded={isMenuOpen}
        aria-label={t('open_action_menu', { name })}
      >
        <MaterialIcon type="more_vert" accessibilityLabel={t('menu')} />
      </button>
    </div>
  )
}

export default FileTreeItemMenu
